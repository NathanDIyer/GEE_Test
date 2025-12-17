# Energy Shape Builder Dashboard

**A tool for generating forecast-quality hourly renewable energy production shapes for any location worldwide.**

## Overview

This dashboard allows users to generate realistic hourly power output profiles for **wind** and **solar** farms at any global location. By leveraging Google Earth Engine's ERA5 reanalysis data, it produces 8,760-hour (1 year) production shapes that can be used for:

- REC (Renewable Energy Certificate) procurement analysis
- Hourly energy matching studies
- Grid integration planning
- Financial modeling of renewable projects
- Carbon accounting with temporal granularity

---

## How It Works

### Data Source

The dashboard pulls meteorological data from **ECMWF ERA5** via Google Earth Engine:

| Resource Type | Dataset                  | Variables Used                                         | Resolution |
| ------------- | ------------------------ | ------------------------------------------------------ | ---------- |
| **Wind**      | `ECMWF/ERA5/HOURLY`      | `u_component_of_wind_100m`, `v_component_of_wind_100m` | ~27km      |
| **Solar**     | `ECMWF/ERA5_LAND/HOURLY` | `surface_solar_radiation_downwards_hourly`             | ~11km      |

### Wind Power Model

The wind model applies a **cubic power curve** to convert 100m wind speeds to power output:

```
Below cut-in speed:     Power = 0
Ramp region:            Power = Rated_MW × ((wind - cut_in) / (rated - cut_in))³
Rated region:           Power = Rated_MW × Max_CF
Above cut-out:          Power = 0 (safety shutdown)
```

**Default Parameters:**

- Cut-in speed: 3.0 m/s
- Rated speed: 12.0 m/s
- Cut-out speed: 25.0 m/s
- Turbine rating: 4.0 MW
- Max plant CF: 90%

The **Max Plant CF** accounts for real-world derating factors:

- Wake losses between turbines (10-20%)
- Turbine availability (~95-97%)
- Spatial wind diversity across plant
- Grid/curtailment constraints

### Solar Power Model

The solar model converts Global Horizontal Irradiance (GHI) to Plane-of-Array (POA) irradiance, then to AC power:

1. **Solar Position**: Calculates sun elevation & azimuth for each hour
2. **GHI → POA**: Accounts for panel tilt, azimuth, and diffuse/direct components
3. **DC Power**: `POA × DC_Capacity × System_Efficiency / 1000`
4. **AC Clipping**: Clips at inverter capacity based on ILR (DC/AC ratio)

**Default Parameters:**

- Panel tilt: 30°
- Panel azimuth: 180° (south-facing)
- Inverter Load Ratio (ILR): 1.25
- System efficiency: 86% (soiling, wiring, temperature losses)

---

## Key Assumptions & Limitations

### Data Limitations

| Limitation                  | Impact                                                 | Mitigation                                             |
| --------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| **Single-point extraction** | Doesn't capture spatial variation across a large plant | Use multiple points and average for large sites        |
| **~27km wind resolution**   | May miss local terrain effects, sea breezes            | Consider site-specific adjustments for complex terrain |
| **Reanalysis data**         | Not actual measurements; may differ from ground truth  | Validate against on-site data where available          |
| **100m hub height only**    | Modern turbines often 120-170m+                        | Apply shear correction for taller turbines             |
| **No wake modeling**        | Plant-level losses approximated via Max CF             | Use dedicated wake models for detailed studies         |

### Model Limitations

| Limitation              | Impact                                          | Mitigation                                  |
| ----------------------- | ----------------------------------------------- | ------------------------------------------- |
| **Generic power curve** | Real turbines have manufacturer-specific curves | Upload custom power curves (future feature) |
| **Simple solar model**  | Isotropic diffuse sky assumption                | Use Perez model for improved accuracy       |
| **No tracking support** | Fixed-tilt only                                 | Adjust effective tilt for 1-axis tracking   |
| **No degradation**      | Assumes Year 1 performance                      | Apply annual degradation factor externally  |
| **No curtailment**      | Doesn't model grid constraints                  | Post-process with curtailment scenarios     |

### Accuracy Expectations

Based on validation against operational plants:

- **Wind CF estimates**: ±5-10% of actual (site-dependent)
- **Solar CF estimates**: ±3-7% of actual
- **Hourly correlation**: R² typically 0.7-0.9 vs. actuals

---

## Ways to Improve (With More Data)

### Near-Term Enhancements

1. **Custom Power Curves**

   - Upload manufacturer-specific turbine curves (IEC 61400-12)
   - Support for multiple turbine types in a plant

2. **Wind Shear Correction**

   - Apply power-law or log-law shear profiles
   - Adjust from 100m to actual hub heights (e.g., 150m)
   - Formula: `V_hub = V_100m × (H_hub / 100)^α` where α ≈ 0.14

3. **Multi-Point Averaging**

   - Sample multiple points across plant footprint
   - Weight by turbine density
   - Captures spatial diversity benefits

4. **Single-Axis Tracking**
   - Implement tracking algorithm for solar
   - Backtracking for GCR optimization

### Advanced Enhancements (Requires Additional Data)

1. **Mesoscale Wind Modeling**

   - Integrate with higher-resolution datasets (e.g., HRRR, WRF)
   - Downscale ERA5 using terrain corrections
   - Better capture of local wind phenomena

2. **Satellite Irradiance**

   - Use NSRDB or Solcast for improved solar estimates
   - Finer temporal resolution (sub-hourly)
   - Better cloud transient modeling

3. **Wake Loss Modeling**

   - Integrate with FLORIS or PyWake
   - Input actual turbine layout
   - Calculate direction-dependent wake losses

4. **Historical Weather Ensemble**

   - Multi-year P50/P90 analysis
   - Capture inter-annual variability
   - Better financing-grade estimates

5. **Grid Integration**

   - Overlay with nodal pricing data
   - Model curtailment based on grid constraints
   - Value-at-risk calculations

6. **Hybrid Plant Modeling**
   - Co-located wind + solar + storage
   - Optimized dispatch strategies
   - Firmed PPA shapes

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface (Dash)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │  Map     │  │ Settings │  │ Progress │  │ Output Charts    ││
│  │(Leaflet) │  │  Panel   │  │   Bar    │  │ (Plotly)         ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Backend (Python/Dash)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Fetch Worker   │  │   Power Model   │  │  Cache Layer    │ │
│  │  (ThreadPool)   │  │  (Wind/Solar)   │  │  (Parquet/CSV)  │ │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘ │
└───────────┼─────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────┐
│              Google Earth Engine API                            │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐ │
│  │  ERA5 Hourly        │  │  ERA5-Land Hourly                │ │
│  │  (Wind @ 100m)      │  │  (Solar Radiation)               │ │
│  └─────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

- **Full year fetch**: ~60-90 seconds (parallelized into 20 chunks)
- **Cached data retrieval**: <1 second
- **Memory footprint**: ~50-100MB per year/location

---

## Output Data Format

### CSV Export Columns (Wind)

| Column            | Description          | Units    |
| ----------------- | -------------------- | -------- |
| `datetime_utc`    | Timestamp            | ISO 8601 |
| `latitude`        | Site latitude        | degrees  |
| `longitude`       | Site longitude       | degrees  |
| `wind_speed_100m` | Wind speed at 100m   | m/s      |
| `power_mw`        | Single-turbine power | MW       |
| `cf`              | Capacity factor      | 0-1      |
| `scaled_mw`       | Scaled to target GWh | MW       |

### CSV Export Columns (Solar)

| Column         | Description                  | Units    |
| -------------- | ---------------------------- | -------- |
| `datetime_utc` | Timestamp                    | ISO 8601 |
| `latitude`     | Site latitude                | degrees  |
| `longitude`    | Site longitude               | degrees  |
| `ghi_wm2`      | Global Horizontal Irradiance | W/m²     |
| `poa_wm2`      | Plane of Array Irradiance    | W/m²     |
| `ac_power_kw`  | AC power output              | kW       |
| `cf`           | Capacity factor              | 0-1      |
| `scaled_mw`    | Scaled to target GWh         | MW       |

---

## Use Cases

### 1. Hourly REC Matching Analysis

```
Target: Match 100 GWh/year load with hourly renewable generation

Steps:
1. Fetch hourly load profile from utility/meter data
2. Generate wind shape for candidate sites
3. Generate solar shape for candidate sites
4. Optimize portfolio weights to maximize hourly match %
5. Export combined shape for 24/7 CFE analysis
```

### 2. Site Comparison

```
Compare multiple candidate sites for a 50 MW wind project:

1. Generate shapes for Site A, B, C
2. Compare:
   - Capacity factors
   - Seasonal profiles (summer vs winter)
   - Diurnal patterns
   - Correlation with load
3. Select optimal site based on value metrics
```

### 3. PPA Structuring

```
Structure a hybrid wind+solar PPA:

1. Generate wind shape (e.g., 60% of portfolio)
2. Generate solar shape (e.g., 40% of portfolio)
3. Combine shapes with target weights
4. Analyze hourly pricing exposure
5. Optimize blend for flat-ish delivery profile
```

---

## Dependencies

```
dash>=2.14.0
dash-bootstrap-components>=1.5.0
dash-leaflet>=0.1.21
dash-extensions>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
earthengine-api>=0.1.350
pyarrow>=12.0.0
```

## Authentication

Requires Google Earth Engine authentication:

```bash
# First time setup
earthengine authenticate

# Or set project explicitly
export EE_PROJECT="your-gcp-project-id"
```

---

## Roadmap

- [ ] Custom turbine power curves (upload CSV)
- [ ] Wind shear correction slider (hub height adjustment)
- [ ] Multi-year analysis (P50/P90)
- [ ] Single-axis tracking for solar
- [ ] Batch mode for multiple locations
- [ ] API endpoint for programmatic access
- [ ] Integration with storage dispatch modeling

---

## Contact

For questions, feature requests, or bug reports, please open an issue or contact the development team.

---

_Note: This tool is intended for screening-level analysis. For investment-grade resource assessments, use site-specific measurements and validated models._
