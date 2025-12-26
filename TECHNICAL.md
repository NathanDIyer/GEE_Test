# Technical Documentation

## Google Earth Engine (GEE) Data Fetching

### Rate Limits & Constraints

| Constraint | Limit |
|------------|-------|
| Max data points per pull | 2,000 (per `getRegion` call) |
| Max API calls per minute | 6,000 (100 calls/second) |

For a full year of hourly data (8,760 hours), we need chunking to stay under the 2,000 data point limit per request.

### Chunking Strategy

```
CHUNK_WORKERS = 20      # Concurrent threads
CHUNK_MULTIPLIER = 3    # Creates smaller chunks
Total chunks = 20 × 3 = 60 chunks per year
```

**Per-chunk breakdown:**
- Hours per chunk: 8,760 / 60 = **146 hours**
- Data points per chunk: 146 hours × 6 bands = **876 points** (well under 2,000 limit)

**Multi-year fetches** process years sequentially to:
- Maximize cache hits (previously fetched years load instantly)
- Provide clear progress feedback ("Year 2023: chunk 15/60")
- Avoid overwhelming GEE with excessive concurrent requests

---

## Data Sources

### Wind Data: ERA5

**Source:** `ECMWF/ERA5/HOURLY`
**Resolution:** ~31km (global reanalysis)

| Band | Unit | Description |
|------|------|-------------|
| `u_component_of_wind_100m` | m/s | East-west wind at 100m |
| `v_component_of_wind_100m` | m/s | North-south wind at 100m |
| `u_component_of_wind_10m` | m/s | East-west wind at 10m |
| `v_component_of_wind_10m` | m/s | North-south wind at 10m |
| `temperature_2m` | K | Air temperature at 2m |
| `surface_pressure` | Pa | Surface pressure |

**Computed bands (server-side on GEE):**

| Computed Band | Formula |
|---------------|---------|
| `wind_speed_100m` | `sqrt(u100² + v100²)` |
| `wind_speed_10m` | `sqrt(u10² + v10²)` |
| `shear_exponent` | `ln(ws100/ws10) / ln(100/10)` |
| `air_density` | `pressure / (287.05 × temperature)` [kg/m³] |

### Solar Data: ERA5-Land

**Source:** `ECMWF/ERA5_LAND/HOURLY`
**Resolution:** ~11km

| Band | Unit | Description |
|------|------|-------------|
| `surface_solar_radiation_downwards_hourly` | J/m² | Accumulated solar radiation |

Converted to Global Horizontal Irradiance (GHI): `GHI = ssrd / 3600` [W/m²]

---

## Wind Power Calculations

### 1. Hub Height Extrapolation (Power Law)

```
ws_hub = ws_100m × (hub_height / 100)^shear_exponent
```

The shear exponent can be:
- **Measured:** Calculated from ERA5 wind speeds at 10m and 100m
- **Manual:** User-specified constant (default: 0.14 for open terrain)

### 2. Power Curve (Cubic Ramp)

```
if ws < cut_in:
    power = 0
elif ws < rated_speed:
    power = rated_MW × ((ws - cut_in) / (rated_speed - cut_in))³
elif ws <= cut_out:
    power = rated_MW × max_plant_cf
else:
    power = 0  # Above cut-out
```

**Default turbine parameters:**
- Cut-in: 3 m/s
- Rated speed: 12 m/s
- Cut-out: 25 m/s
- Rated power: 3 MW
- Max plant CF: 90%

### 3. Air Density Correction (Optional)

```
P_adjusted = P × (air_density / 1.225)
```

Standard air density = **1.225 kg/m³** at sea level, 15°C

Higher elevations and warmer temperatures reduce air density, which reduces power output.

### 4. Capacity Factor

```
CF = power_mw / rated_mw
```

### 5. Scaling to Target GWh

```
installed_mw = target_gwh × 1000 / (8760 × avg_cf)
scaled_mw = cf × installed_mw
```

---

## Solar Power Calculations

### 1. Solar Position

**Declination angle** (seasonal tilt of Earth):
```
declination = 23.45° × sin(360/365 × (284 + day_of_year))
```

**Hour angle** (time relative to solar noon):
```
solar_time = hour + longitude / 15
hour_angle = 15° × (solar_time - 12)
```

**Solar elevation**:
```
sin(elevation) = sin(lat)×sin(dec) + cos(lat)×cos(dec)×cos(hour_angle)
```

### 2. GHI to POA (Plane of Array) Conversion

The model splits irradiance into three components:

| Component | Fraction | Description |
|-----------|----------|-------------|
| Direct | ~80% | Adjusted for panel tilt and sun angle |
| Diffuse | ~20% | Isotropic sky model |
| Ground-reflected | ~2-4% | Albedo = 0.2 |

```
POA = direct_poa + diffuse + ground_reflected
```

Capped at 1,300 W/m² (clear sky maximum on tilted surface).

### 3. DC Power

```
dc_power_kw = POA × dc_capacity_kw × efficiency / 1000
```

Default system efficiency: **86%** (accounts for soiling, wiring losses, temperature)

### 4. AC Power (Inverter Clipping)

```
ac_capacity_kw = dc_capacity_kw / ILR
ac_power_kw = min(dc_power_kw, ac_capacity_kw)
```

**ILR (Inverter Load Ratio):** DC/AC capacity ratio, typically 1.2-1.4

---

## Curtailment Analysis

The curtailment feature analyzes the trade-off between hourly matching and energy retention.

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Cap Level** | User input (0-100%) | Maximum production as % of peak |
| **Curtailed CF** | `min(cf, cap_fraction)` | CF after applying cap |
| **Hourly Matching %** | `sum(curtailed_cf) / (cap × hours) × 100` | How "baseload-like" the profile is |
| **Energy Retained %** | `mean(curtailed_cf) / mean(cf) × 100` | Energy kept vs. thrown away |

### Interpretation

- **At 100% cap (no curtailment):**
  - Matching % = Capacity Factor (e.g., ~30% for wind)
  - Energy Retained = 100%

- **As cap decreases:**
  - Matching % increases (profile becomes more baseload-like)
  - Energy Retained decreases (more generation is curtailed)

- **At very low cap:**
  - Matching % approaches 100% (wind almost always exceeds the low cap)
  - Energy Retained drops significantly

### Use Case

This analysis helps evaluate scenarios where a buyer wants more consistent hourly delivery (higher matching) but must sacrifice total energy volume. The curve shows the exact trade-off at each cap level.

---

## Caching

Fetched data is cached locally to avoid re-fetching:

```
data/era5_{year}_lat_{lat}_lon_{lon}.parquet
data/era5_solar_{year}_lat_{lat}_lon_{lon}.parquet
```

- Coordinates rounded to 2 decimal places (~1km precision)
- Parquet format preferred; falls back to CSV if pyarrow unavailable
- Multi-year fetches benefit significantly from caching

---

## Statistics & Quality Metrics

### Autocorrelation
- **1-hour lag:** Hour-to-hour persistence
- **24-hour lag:** Daily rhythm strength

### Low CF Analysis
- **Low CF hours %:** Share of hours below 10% CF
- **Max low streak:** Longest consecutive run below 10% CF

### Variability
- **P95 ramp:** 95th percentile of hour-to-hour MW changes
- **Weibull fit:** Shape (k) and scale (λ) parameters for wind speed distribution

### Multi-Year Statistics
- **Inter-annual CF variability:** Standard deviation of annual capacity factors
- **Weibull k range:** Min/max shape parameters across years
