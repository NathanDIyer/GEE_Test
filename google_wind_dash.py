import math
import os
import re
import threading
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
from io import StringIO
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
import ee

# Password protection
from dash_auth import BasicAuth

# See TECHNICAL.md for detailed documentation on GEE data fetching and calculations

CHUNK_HOURS = None  # computed dynamically per year
CHUNK_WORKERS = 20  # number of concurrent chunk fetches
CHUNK_MULTIPLIER = 3  # multiplier for chunk count (more chunks = smaller requests, same parallelism)

# Try to import dash_extensions for proper eventHandlers support
try:
    from dash_extensions.javascript import assign
    marker_dragend_js = dict(
        dragend=assign("function(e, ctx) { ctx.setProps({data: e.target.getLatLng()}) }")
    )
    print("[INFO] dash-extensions loaded - marker drag events enabled")
except ImportError:
    marker_dragend_js = {}
    print("[WARNING] dash-extensions not installed - marker drag won't update inputs")
    print("[WARNING] Install with: pip install dash-extensions")

# ---- Earth Engine helpers ----


def init_ee():
    """Initialize Earth Engine - supports local credentials or service account for deployment."""
    import json
    import tempfile

    project = os.environ.get("EE_PROJECT")
    sa_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

    if sa_creds_json:
        # Service account credentials provided as JSON string (for Render/cloud deployment)
        try:
            # Parse to validate and extract email
            sa_creds = json.loads(sa_creds_json)
            service_email = sa_creds.get("client_email")
            print(f"[GEE] Attempting service account auth for: {service_email}")
            print(f"[GEE] Project: {project}")

            # Write to temp file to avoid any string escaping issues
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sa_creds, f)
                key_file_path = f.name
            print(f"[GEE] Wrote credentials to temp file: {key_file_path}")

            credentials = ee.ServiceAccountCredentials(
                service_email,
                key_file=key_file_path
            )
            ee.Initialize(credentials=credentials, project=project)
            print(f"[GEE] Successfully initialized with service account: {service_email}")
            return
        except json.JSONDecodeError as exc:
            print(f"[GEE] FATAL: Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {exc}")
            print(f"[GEE] First 100 chars of JSON: {sa_creds_json[:100] if sa_creds_json else 'empty'}")
            raise
        except Exception as exc:
            print(f"[GEE] FATAL: Service account init failed: {exc}")
            print("[GEE] Make sure the service account is registered with Earth Engine at:")
            print("[GEE] https://signup.earthengine.google.com or via code.earthengine.google.com")
            raise

    # Local development - use default credentials
    if project:
        ee.Initialize(project=project)
        print(f"[GEE] Initialized with project: {project}")
    else:
        ee.Initialize()
        print("[GEE] Initialized with default credentials")


def add_wind_metrics(image):
    """Compute wind speed, shear exponent, and air density on GEE server side."""
    # Wind speed at 100m
    u100 = image.select("u_component_of_wind_100m")
    v100 = image.select("v_component_of_wind_100m")
    ws100 = u100.hypot(v100).rename("wind_speed_100m")

    # Wind speed at 10m
    u10 = image.select("u_component_of_wind_10m")
    v10 = image.select("v_component_of_wind_10m")
    ws10 = u10.hypot(v10).rename("wind_speed_10m")

    # Shear exponent: α = ln(ws100/ws10) / ln(100/10) = ln(ws100/ws10) / 2.303
    # Clamp ws10 to avoid division by zero (min 0.1 m/s)
    ws10_safe = ws10.max(0.1)
    shear = ws100.divide(ws10_safe).log().divide(2.302585).rename("shear_exponent")

    # Air density: ρ = P / (R * T) where R = 287.05 J/(kg·K)
    # ERA5 temperature_2m is in Kelvin, surface_pressure is in Pa
    temp = image.select("temperature_2m")
    pressure = image.select("surface_pressure")
    density = pressure.divide(temp.multiply(287.05)).rename("air_density")

    return image.addBands([ws100, ws10, shear, density]).copyProperties(image, image.propertyNames())


def extrapolate_wind_speed(ws_100m, hub_height, shear_exponent=0.14):
    """Extrapolate wind speed from 100m to hub height using power law.

    Formula: ws_hub = ws_100m * (hub_height / 100)^shear_exponent

    Args:
        ws_100m: Wind speed at 100m (m/s), can be scalar or array
        hub_height: Target hub height in meters
        shear_exponent: Wind shear exponent (default 0.14 for open terrain)

    Returns:
        Extrapolated wind speed at hub height
    """
    if hub_height == 100:
        return ws_100m
    return ws_100m * (hub_height / 100) ** shear_exponent


# Standard air density at sea level, 15°C (kg/m³)
STANDARD_AIR_DENSITY = 1.225


def adjust_power_for_density(power, air_density, apply_correction=True):
    """Adjust power output for air density variations.

    Wind turbine power is proportional to air density: P ∝ ½ρAv³
    This function scales power by the ratio of actual to standard density.

    Args:
        power: Power output (MW), can be scalar or array
        air_density: Actual air density (kg/m³), can be scalar or array
        apply_correction: If False, returns power unchanged

    Returns:
        Density-adjusted power output
    """
    if not apply_correction:
        return power
    return power * (air_density / STANDARD_AIR_DENSITY)


def fetch_era5_slice(lat, lon, start_iso, hours):
    """Pull a small ERA5 slice and return a sorted DataFrame."""
    start = pd.to_datetime(start_iso, utc=True)
    end = start + pd.to_timedelta(hours, unit="h")
    point = ee.Geometry.Point([lon, lat])

    # Select all bands needed for wind speed, shear, and density calculations
    raw_bands = [
        "u_component_of_wind_100m",
        "v_component_of_wind_100m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "temperature_2m",
        "surface_pressure",
    ]
    # Computed bands from add_wind_metrics
    computed_bands = ["wind_speed_100m", "wind_speed_10m", "shear_exponent", "air_density"]

    coll = (
        ee.ImageCollection("ECMWF/ERA5/HOURLY")
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
        .select(raw_bands)
        .map(add_wind_metrics)
        .select(computed_bands)
    )

    data = coll.getRegion(point, 27000).getInfo()
    if len(data) <= 1:
        return pd.DataFrame()

    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    keep = [
        c
        for c in [
            "datetime",
            "longitude",
            "latitude",
            "wind_speed_100m",
            "wind_speed_10m",
            "shear_exponent",
            "air_density",
        ]
        if c in df.columns
    ]
    return df[keep].sort_values("datetime").reset_index(drop=True)


def fetch_era5_year_chunked(
    lat,
    lon,
    year=2024,
    progress_hook=None,
    chunk_hours=CHUNK_HOURS,
    max_workers=CHUNK_WORKERS,
):
    """Chunked fetch for a full year (parallelized).

    Uses CHUNK_MULTIPLIER to create more chunks than workers, keeping each
    request smaller while maintaining the same level of parallelism.
    """
    start_iso = f"{year}-01-01T00:00:00Z"
    end_iso = f"{year + 1}-01-01T00:00:00Z"
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)
    worker_count = max_workers or CHUNK_WORKERS
    worker_count = max(1, worker_count)

    # Create more chunks than workers to keep each request smaller
    # (compensates for fetching more bands: shear + density data)
    num_chunks = worker_count * CHUNK_MULTIPLIER
    total_hours = (end - start) / pd.Timedelta(hours=1)
    chunk_delta = (end - start) / num_chunks

    chunks = []
    for idx in range(num_chunks):
        chunk_start_dt = start + idx * chunk_delta
        chunk_end_dt = end if idx == num_chunks - 1 else start + (idx + 1) * chunk_delta
        hours = (chunk_end_dt - chunk_start_dt) / pd.Timedelta(hours=1)
        chunks.append((idx, chunk_start_dt.isoformat(), hours))

    total_chunks = num_chunks
    if progress_hook:
        progress_hook(0, total_chunks)

    if not chunks:
        return pd.DataFrame(), total_chunks

    def fetch_one(chunk_meta):
        chunk_idx, chunk_start, hours = chunk_meta
        df_chunk = fetch_era5_slice(lat, lon, chunk_start, hours)
        return chunk_idx, df_chunk

    frames = [None] * total_chunks
    completed = 0
    # Still use worker_count for parallelism (not num_chunks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_one, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk_idx, df_chunk = future.result()
            frames[chunk_idx] = df_chunk
            completed += 1
            if progress_hook:
                progress_hook(completed, total_chunks)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), total_chunks

    combined = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    return combined, total_chunks


def apply_power_curve(ws_series, cut_in, rated_speed, cut_out, rated_power_mw, max_cf=0.90):
    """
    Realistic wind power curve with plant-level constraints.

    Args:
        ws_series: Wind speed array (m/s)
        cut_in: Minimum wind speed for generation (m/s)
        rated_speed: Wind speed at which turbine reaches rated power (m/s)
        cut_out: Maximum operating wind speed (m/s)
        rated_power_mw: Turbine nameplate capacity (MW)
        max_cf: Maximum plant capacity factor (0-1), accounts for wake losses,
                availability, and spatial diversity. Real wind farms rarely
                exceed 85-95% of nameplate even in ideal conditions.

    Returns:
        Array of power output (MW)
    """
    ws = np.asarray(ws_series, dtype=float)
    power = np.zeros_like(ws, dtype=float)

    # Cubic ramp from cut-in to rated speed
    mask_ramp = (ws >= cut_in) & (ws < rated_speed)
    power[mask_ramp] = rated_power_mw * (
        (ws[mask_ramp] - cut_in) / (rated_speed - cut_in)
    ) ** 3

    # Rated power region (capped by plant-level max CF)
    mask_rated = (ws >= rated_speed) & (ws <= cut_out)
    power[mask_rated] = rated_power_mw

    # Apply plant-level capacity factor cap to account for:
    # - Wake losses between turbines (10-20%)
    # - Turbine availability (~95-97%)
    # - Spatial wind diversity across plant
    # - Grid/curtailment constraints
    max_power = rated_power_mw * max_cf
    power = np.minimum(power, max_power)

    return power


# ---- Solar helpers ----


def fetch_era5_solar_slice(lat, lon, start_iso, hours):
    """Pull ERA5 solar radiation data for a time slice."""
    start = pd.to_datetime(start_iso, utc=True)
    end = start + pd.to_timedelta(hours, unit="h")
    point = ee.Geometry.Point([lon, lat])

    coll = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
        .select(["surface_solar_radiation_downwards_hourly"])
    )

    data = coll.getRegion(point, 11132).getInfo()  # ~11km resolution for ERA5-Land
    if len(data) <= 1:
        return pd.DataFrame()

    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    # ERA5-Land ssrd is in J/m² accumulated over the hour, convert to W/m² (average)
    if "surface_solar_radiation_downwards_hourly" in df.columns:
        df["ghi_wm2"] = df["surface_solar_radiation_downwards_hourly"] / 3600.0

    keep = [c for c in ["datetime", "longitude", "latitude", "ghi_wm2"] if c in df.columns]
    return df[keep].sort_values("datetime").reset_index(drop=True)


def fetch_era5_solar_year_chunked(
    lat,
    lon,
    year=2024,
    progress_hook=None,
    chunk_hours=CHUNK_HOURS,
    max_workers=CHUNK_WORKERS,
):
    """Chunked fetch for a full year of solar radiation data (parallelized)."""
    start_iso = f"{year}-01-01T00:00:00Z"
    end_iso = f"{year + 1}-01-01T00:00:00Z"
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)
    worker_count = max_workers or CHUNK_WORKERS
    worker_count = max(1, worker_count)

    total_hours = (end - start) / pd.Timedelta(hours=1)
    chunk_delta = (end - start) / worker_count

    chunks = []
    for idx in range(worker_count):
        chunk_start_dt = start + idx * chunk_delta
        chunk_end_dt = end if idx == worker_count - 1 else start + (idx + 1) * chunk_delta
        hours = (chunk_end_dt - chunk_start_dt) / pd.Timedelta(hours=1)
        chunks.append((idx, chunk_start_dt.isoformat(), hours))

    total_chunks = worker_count
    if progress_hook:
        progress_hook(0, total_chunks)

    if not chunks:
        return pd.DataFrame(), total_chunks

    def fetch_one(chunk_meta):
        chunk_idx, chunk_start, hours = chunk_meta
        df_chunk = fetch_era5_solar_slice(lat, lon, chunk_start, hours)
        return chunk_idx, df_chunk

    frames = [None] * total_chunks
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(fetch_one, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk_idx, df_chunk = future.result()
            frames[chunk_idx] = df_chunk
            completed += 1
            if progress_hook:
                progress_hook(completed, total_chunks)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), total_chunks

    combined = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    return combined, total_chunks


def calculate_solar_position(datetime_utc, lat, lon):
    """Calculate solar position (elevation, azimuth) for a datetime array."""
    # Convert to day of year and hour
    doy = datetime_utc.dt.dayofyear.values
    hour = datetime_utc.dt.hour.values + datetime_utc.dt.minute.values / 60.0

    # Solar declination (degrees)
    declination = 23.45 * np.sin(np.radians(360 / 365 * (284 + doy)))

    # Hour angle (degrees) - solar noon = 0
    # Approximate: assume longitude corresponds to time zone offset
    solar_time = hour + lon / 15.0
    hour_angle = 15.0 * (solar_time - 12.0)

    # Solar elevation angle
    lat_rad = np.radians(lat)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    sin_elev = (np.sin(lat_rad) * np.sin(dec_rad) +
                np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
    elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))

    # Solar azimuth (degrees from north, clockwise)
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_elev) / (np.cos(lat_rad) * np.cos(np.radians(elevation)) + 1e-10)
    azimuth = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    # Correct azimuth for afternoon (hour angle > 0)
    azimuth = np.where(hour_angle > 0, 360 - azimuth, azimuth)

    return elevation, azimuth


def ghi_to_poa(ghi, solar_elevation, solar_azimuth, panel_tilt, panel_azimuth):
    """
    Convert Global Horizontal Irradiance to Plane of Array irradiance.
    Simple model assuming mostly direct normal irradiance.

    Parameters:
    - ghi: Global Horizontal Irradiance (W/m²)
    - solar_elevation: Sun elevation angle (degrees)
    - solar_azimuth: Sun azimuth (degrees from north)
    - panel_tilt: Panel tilt from horizontal (degrees)
    - panel_azimuth: Panel azimuth (degrees from north, 180=south)
    """
    ghi = np.asarray(ghi, dtype=float)

    # When sun is below horizon, no irradiance
    mask_day = solar_elevation > 0
    poa = np.zeros_like(ghi)

    # Convert angles to radians
    elev_rad = np.radians(solar_elevation)
    sol_az_rad = np.radians(solar_azimuth)
    tilt_rad = np.radians(panel_tilt)
    pan_az_rad = np.radians(panel_azimuth)

    # Angle of incidence on tilted surface
    cos_aoi = (np.sin(elev_rad) * np.cos(tilt_rad) +
               np.cos(elev_rad) * np.sin(tilt_rad) * np.cos(sol_az_rad - pan_az_rad))
    cos_aoi = np.clip(cos_aoi, 0, 1)

    # Simple model: POA = GHI * (cos(AOI) / sin(elevation)) for direct, plus diffuse
    # Use a diffuse fraction estimate (~20% at clear sky)
    diffuse_frac = 0.2
    sin_elev = np.sin(elev_rad)
    sin_elev = np.where(sin_elev < 0.05, 0.05, sin_elev)  # Avoid division issues at low sun

    # Direct component adjusted for tilt
    direct_horizontal = ghi * (1 - diffuse_frac)
    direct_poa = direct_horizontal * cos_aoi / sin_elev

    # Diffuse component (isotropic sky model)
    diffuse = ghi * diffuse_frac * (1 + np.cos(tilt_rad)) / 2

    # Ground reflected (albedo ~0.2)
    ground_reflected = ghi * 0.2 * (1 - np.cos(tilt_rad)) / 2

    poa[mask_day] = (direct_poa + diffuse + ground_reflected)[mask_day]

    # Cap POA at reasonable maximum (clear sky ~1100 W/m² max on tilted surface)
    poa = np.clip(poa, 0, 1300)

    return poa


def apply_solar_power(ghi_series, datetime_series, lat, lon, panel_tilt, panel_azimuth,
                      ilr, dc_capacity_kw, system_efficiency=0.86):
    """
    Calculate AC power output from a solar array.

    Parameters:
    - ghi_series: Global Horizontal Irradiance (W/m²)
    - datetime_series: Timestamps (UTC)
    - lat, lon: Location
    - panel_tilt: Panel tilt angle (degrees)
    - panel_azimuth: Panel azimuth (degrees, 180=south)
    - ilr: Inverter Load Ratio (DC/AC capacity ratio)
    - dc_capacity_kw: DC nameplate capacity (kW)
    - system_efficiency: Overall system efficiency (default 0.86 accounts for soiling, wiring, temp)

    Returns:
    - ac_power_kw: AC power output clipped by inverter capacity
    """
    # Calculate solar position
    elevation, azimuth = calculate_solar_position(datetime_series, lat, lon)

    # Convert GHI to POA irradiance
    poa = ghi_to_poa(ghi_series.values, elevation, azimuth, panel_tilt, panel_azimuth)

    # DC power = POA * DC capacity * efficiency / 1000 (reference irradiance)
    # Reference irradiance is 1000 W/m² (STC conditions)
    dc_power_kw = poa * dc_capacity_kw * system_efficiency / 1000.0

    # AC capacity based on ILR
    ac_capacity_kw = dc_capacity_kw / ilr

    # Clip at inverter capacity (AC output)
    ac_power_kw = np.minimum(dc_power_kw, ac_capacity_kw)

    return ac_power_kw, poa


def log_gee_error(context: str, exc: Exception):
    traceback.print_exc()
    print(f"[GEE] {context}: {exc}")


# ---- Caching helpers ----

CACHE_DIR = "data"


def get_cache_path(lat, lon, year=2024, use_parquet=True, energy_type="wind"):
    """Generate cache file path for a location."""
    # Round to 2 decimal places for cache key (about 1km precision)
    lat_key = round(lat, 2)
    lon_key = round(lon, 2)
    os.makedirs(CACHE_DIR, exist_ok=True)
    ext = "parquet" if use_parquet else "csv"
    prefix = "era5_solar" if energy_type == "solar" else "era5"
    return os.path.join(CACHE_DIR, f"{prefix}_{year}_lat_{lat_key}_lon_{lon_key}.{ext}")


def load_from_cache(lat, lon, year=2024, energy_type="wind"):
    """Load cached data for a location if it exists."""
    # Try parquet first
    cache_path = get_cache_path(lat, lon, year, use_parquet=True, energy_type=energy_type)
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"[Cache] Loaded {len(df)} rows from cache: {cache_path}")
            return df
        except Exception as exc:
            print(f"[Cache] Parquet read failed, trying CSV: {exc}")

    # Fall back to CSV
    cache_path = get_cache_path(lat, lon, year, use_parquet=False, energy_type=energy_type)
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            print(f"[Cache] Loaded {len(df)} rows from cache: {cache_path}")
            return df
        except Exception as exc:
            print(f"[Cache] Error loading cache: {exc}")
            return None
    return None


def save_to_cache(df, lat, lon, year=2024, energy_type="wind"):
    """Save DataFrame to cache."""
    if df.empty:
        return
    # Try parquet first, fall back to CSV
    cache_path = get_cache_path(lat, lon, year, use_parquet=True, energy_type=energy_type)
    try:
        df.to_parquet(cache_path, index=False)
        print(f"[Cache] Saved {len(df)} rows to cache: {cache_path}")
    except Exception as exc:
        print(f"[Cache] Parquet save failed, using CSV: {exc}")
        cache_path = get_cache_path(lat, lon, year, use_parquet=False, energy_type=energy_type)
        try:
            df.to_csv(cache_path, index=False)
            print(f"[Cache] Saved {len(df)} rows to cache: {cache_path}")
        except Exception as exc2:
            print(f"[Cache] Error saving cache: {exc2}")


def get_cached_locations(year=2024, energy_type="wind"):
    """Scan cache directory and return list of (lat, lon) tuples for cached locations."""
    cached = []
    if not os.path.exists(CACHE_DIR):
        return cached
    prefix = "era5_solar" if energy_type == "solar" else "era5"
    pattern = re.compile(rf"{prefix}_{year}_lat_([-\d.]+)_lon_([-\d.]+)\.(parquet|csv)")
    for filename in os.listdir(CACHE_DIR):
        match = pattern.match(filename)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            cached.append((lat, lon))
    return cached


# ---- Fetch state (polled by progress interval) ----

FETCH_STATE = {
    "status": "idle",
    "progress": 0,
    "total": 1,
    "message": "Idle",
    "result": None,
    "last_result": None,
    "error": None,
    "duration": 0.0,
    # Multi-year mode fields
    "current_year": None,
    "years_completed": [],
    "years_pending": [],
    "multi_year_results": {},
}
FETCH_LOCK = threading.Lock()


def set_fetch_state(**kwargs):
    with FETCH_LOCK:
        FETCH_STATE.update(**kwargs)


def get_fetch_state():
    with FETCH_LOCK:
        return dict(FETCH_STATE)


def fetch_worker(lat, lon, year=2024, energy_type="wind"):
    set_fetch_state(status="running", progress=0, total=1, message="Starting...")
    started = time.perf_counter()
    try:
        # Check cache first
        df = load_from_cache(lat, lon, year=year, energy_type=energy_type)
        if df is not None and not df.empty:
            duration = time.perf_counter() - started
            payload = {
                "data": df.to_json(date_format="iso", orient="split"),
                "lat": lat,
                "lon": lon,
                "year": year,
                "energy_type": energy_type,
                "ts": time.time(),
            }
            set_fetch_state(
                status="done",
                progress=1,
                total=1,
                message=f"Loaded from cache in {duration:.2f}s",
                result=payload,
                error=None,
                duration=duration,
            )
            return

        # Not in cache, fetch from Earth Engine
        def hook(idx, total):
            set_fetch_state(
                status="running",
                progress=idx,
                total=total,
                message=f"Chunk {idx}/{total}",
            )

        if energy_type == "solar":
            df, total_chunks = fetch_era5_solar_year_chunked(
                lat, lon, year=year, progress_hook=hook, chunk_hours=CHUNK_HOURS
            )
        else:
            df, total_chunks = fetch_era5_year_chunked(
                lat, lon, year=year, progress_hook=hook, chunk_hours=CHUNK_HOURS
            )

        # Save to cache after successful fetch
        if not df.empty:
            save_to_cache(df, lat, lon, year=year, energy_type=energy_type)

        duration = time.perf_counter() - started
        payload = {
            "data": df.to_json(date_format="iso", orient="split"),
            "lat": lat,
            "lon": lon,
            "year": year,
            "energy_type": energy_type,
            "ts": time.time(),
        }
        set_fetch_state(
            status="done",
            progress=total_chunks,
            total=total_chunks,
            message=f"Done in {duration:.1f}s",
            result=payload,
            error=None,
            duration=duration,
        )
    except Exception as exc:
        log_gee_error("fetch_worker", exc)
        set_fetch_state(status="error", error=str(exc), message=str(exc))


def fetch_multi_year_worker(lat, lon, years, energy_type="wind"):
    """
    Fetch multiple years sequentially, leveraging per-year caching.

    Fetches years one at a time to maximize cache hits (previously fetched years
    load instantly) while keeping 20 workers available for chunk parallelism
    within each year.
    """
    set_fetch_state(
        status="running",
        progress=0,
        total=len(years),
        message=f"Starting multi-year fetch ({len(years)} years)...",
        current_year=years[0] if years else None,
        years_completed=[],
        years_pending=list(years),
        multi_year_results={},
    )

    started = time.perf_counter()
    results = {}

    for idx, year in enumerate(years):
        set_fetch_state(
            current_year=year,
            message=f"Fetching {year} ({idx + 1}/{len(years)})...",
        )

        try:
            # Check cache first
            df = load_from_cache(lat, lon, year=year, energy_type=energy_type)

            if df is None or df.empty:
                # Not cached, fetch from GEE
                def chunk_hook(chunk_idx, chunk_total):
                    set_fetch_state(
                        message=f"Year {year}: chunk {chunk_idx}/{chunk_total}",
                    )

                if energy_type == "solar":
                    df, _ = fetch_era5_solar_year_chunked(
                        lat, lon, year=year, progress_hook=chunk_hook, chunk_hours=CHUNK_HOURS
                    )
                else:
                    df, _ = fetch_era5_year_chunked(
                        lat, lon, year=year, progress_hook=chunk_hook, chunk_hours=CHUNK_HOURS
                    )

                # Save to cache
                if df is not None and not df.empty:
                    save_to_cache(df, lat, lon, year=year, energy_type=energy_type)

            # Store result
            if df is not None and not df.empty:
                payload = {
                    "data": df.to_json(date_format="iso", orient="split"),
                    "lat": lat,
                    "lon": lon,
                    "year": year,
                    "energy_type": energy_type,
                    "ts": time.time(),
                }
                results[year] = payload

                # Update state with completed year
                with FETCH_LOCK:
                    FETCH_STATE["years_completed"].append(year)
                    if year in FETCH_STATE["years_pending"]:
                        FETCH_STATE["years_pending"].remove(year)
                    FETCH_STATE["multi_year_results"][year] = payload
                    FETCH_STATE["progress"] = idx + 1

        except Exception as exc:
            log_gee_error(f"fetch_multi_year_worker year {year}", exc)
            # Continue with other years, mark this one as failed
            set_fetch_state(message=f"Year {year} failed: {exc}, continuing...")

    duration = time.perf_counter() - started
    set_fetch_state(
        status="done",
        progress=len(years),
        total=len(years),
        message=f"Fetched {len(results)}/{len(years)} years in {duration:.1f}s",
        result=results,  # Dict of {year: payload}
        error=None,
        duration=duration,
    )


# ---- Dash app ----


init_ee()

# Clear any stale fetch result on startup so a fresh session never auto-loads old data
set_fetch_state(
    status="idle",
    progress=0,
    total=1,
    message="Idle",
    result=None,
    last_result=None,
    error=None,
    duration=0.0,
)

google_colors = {
    "blue": "#4285F4",
    "green": "#34A853",
    "yellow": "#FBBC05",
    "red": "#EA4335",
    "gray": "#f8f9fa",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&family=Roboto:wght@400;500;600;700&display=swap",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css",
    ],
)
app.title = "Energy Shape Builder"
server = app.server

# Set secret key for session management (required for auth)
server.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# Password protection - set DASH_USERNAME and DASH_PASSWORD environment variables
# For local dev without auth, these default to None which disables auth
dash_user = os.environ.get("DASH_USERNAME")
dash_pass = os.environ.get("DASH_PASSWORD")
if dash_user and dash_pass:
    auth = BasicAuth(app, {dash_user: dash_pass})
    print(f"[Auth] Password protection enabled for user: {dash_user}")
else:
    print("[Auth] No DASH_USERNAME/DASH_PASSWORD set - running without auth")

default_lat = 39.8283
default_lon = -98.5795

app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H2(
                                                    id="main-title",
                                                    children="Forecast-quality shapes for your wind farm",
                                                    style={
                                                        "color": "#202124",
                                                        "fontWeight": "700",
                                                        "marginBottom": "6px",
                                                    },
                                                ),
                                                html.P(
                                                    id="main-description",
                                                    children="Drop a pin for location, choose how many RECs you purchase, tweak advanced turbine settings, then fetch. Outputs and CSV download unlock as soon as the run finishes.",
                                                    style={"color": "#3c4043", "marginBottom": 0},
                                                ),
                                            ],
                                            md=9,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Energy Type", style={"fontWeight": "600", "color": "#202124"}),
                                                    dcc.Dropdown(
                                                        id="energy-type-dropdown",
                                                        options=[
                                                            {"label": "Solar", "value": "solar"},
                                                            {"label": "Wind", "value": "wind"},
                                                        ],
                                                        value="wind",
                                                        clearable=False,
                                                        style={"fontWeight": "500"},
                                                    ),
                                            ],
                                            md=3,
                                            style={"display": "flex", "flexDirection": "column", "justifyContent": "center"},
                                        ),
                                    ],
                                    align="center",
                                ),
                            ]
                        )
                    ],
                    className="mb-4 shadow-sm",
                    style={
                        "border": "none",
                        "background": "linear-gradient(120deg, #e8f0fe 0%, #f8fbff 50%, #e0f3ec 100%)",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            id="location-card-header",
                                            children="Where is your wind farm?",
                                            style={
                                                "fontWeight": "700",
                                                "color": google_colors["blue"],
                                                "backgroundColor": "#fff",
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                dl.Map(
                                                    id="location-map",
                                                    center=[default_lat, default_lon],
                                                    zoom=3,
                                                    style={
                                                        "width": "100%",
                                                        "height": "330px",
                                                        "borderRadius": "8px",
                                                        "overflow": "hidden",
                                                    },
                                                    children=[
                                                        dl.TileLayer(),
                                                        dl.LayerGroup(id="cached-markers"),
                                                        dl.Marker(
                                                            id="location-marker",
                                                            position=[default_lat, default_lon],
                                                            draggable=True,
                                                            eventHandlers=marker_dragend_js,
                                                        ),
                                                    ],
                                                ),
                                                html.Small(
                                                    "Drag the marker or click the map to pick your site. Cached locations show as green dots.",
                                                    style={"color": "#5f6368"},
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label("Latitude"),
                                                                        dbc.Input(
                                                                            id="lat-input",
                                                                            type="number",
                                                                            value=default_lat,
                                                                            step=0.1,
                                                                        ),
                                                                    ]
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Label("Longitude"),
                                                                        dbc.Input(
                                                                            id="lon-input",
                                                                            type="number",
                                                                            value=default_lon,
                                                                            step=0.1,
                                                                        ),
                                                                    ]
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        # Single year dropdown (visible when not in multi-year mode)
                                                                        html.Div(
                                                                            id="single-year-container",
                                                                            children=[
                                                                                dbc.Label("Year"),
                                                                                dcc.Dropdown(
                                                                                    id="year-dropdown",
                                                                                    options=[
                                                                                        {"label": str(y), "value": y}
                                                                                        for y in range(2024, 2009, -1)
                                                                                    ],
                                                                                    value=2024,
                                                                                    clearable=False,
                                                                                    style={"minWidth": "90px"},
                                                                                ),
                                                                            ],
                                                                        ),
                                                                        # Multi-year range picker (hidden by default)
                                                                        html.Div(
                                                                            id="year-range-container",
                                                                            style={"display": "none"},
                                                                            children=[
                                                                                dbc.Label("Year Range"),
                                                                                dbc.Row(
                                                                                    [
                                                                                        dbc.Col(
                                                                                            dcc.Dropdown(
                                                                                                id="start-year-dropdown",
                                                                                                options=[
                                                                                                    {"label": str(y), "value": y}
                                                                                                    for y in range(2024, 2009, -1)
                                                                                                ],
                                                                                                value=2015,
                                                                                                clearable=False,
                                                                                                placeholder="Start",
                                                                                            ),
                                                                                            width=6,
                                                                                        ),
                                                                                        dbc.Col(
                                                                                            dcc.Dropdown(
                                                                                                id="end-year-dropdown",
                                                                                                options=[
                                                                                                    {"label": str(y), "value": y}
                                                                                                    for y in range(2024, 2009, -1)
                                                                                                ],
                                                                                                value=2024,
                                                                                                clearable=False,
                                                                                                placeholder="End",
                                                                                            ),
                                                                                            width=6,
                                                                                        ),
                                                                                    ],
                                                                                    className="g-1",
                                                                                ),
                                                                                html.Div(
                                                                                    id="year-range-validation",
                                                                                    style={"color": "#EA4335", "fontSize": "11px", "marginTop": "2px"},
                                                                                ),
                                                                            ],
                                                                        ),
                                                                        # Multi-year toggle
                                                                        dbc.Switch(
                                                                            id="multi-year-toggle",
                                                                            label="Multi-year",
                                                                            value=False,
                                                                            style={"marginTop": "6px", "fontSize": "12px"},
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ],
                                                            className="mt-3",
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="h-100 shadow-sm",
                                    style={"border": "1px solid #e0e0e0"},
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            "How many RECs do you purchase?",
                                            style={
                                                "fontWeight": "700",
                                                "color": google_colors["green"],
                                                "backgroundColor": "#fff",
                                            },
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Annual purchase (GWh, scales the hourly profile)",
                                                            style={"fontWeight": "500"},
                                                        ),
                                                        html.Div(
                                                            id="target-mwh-display",
                                                            style={"color": "#5f6368", "fontSize": "14px"},
                                                        ),
                                                    ],
                                                    className="d-flex justify-content-between align-items-center mb-2",
                                                ),
                                                dbc.Input(
                                                    id="target-mwh",
                                                    type="number",
                                                    min=0,
                                                    step=10,
                                                    value=100,
                                                    placeholder="Enter annual GWh",
                                                    style={"maxWidth": "160px"},
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            "Fetch profile",
                                                            id="fetch-btn",
                                                            size="lg",
                                                            color="primary",
                                                            style={
                                                                "backgroundColor": google_colors["blue"],
                                                                "borderColor": google_colors["blue"],
                                                                "fontWeight": "600",
                                                            },
                                                            className="mt-3",
                                                        ),
                                                        html.Div(
                                                            id="status-text",
                                                            style={"marginTop": "12px", "color": "#3c4043"},
                                                        ),
                                                        dbc.Progress(
                                                            id="progress-bar",
                                                            value=0,
                                                            label="Idle",
                                                            striped=False,
                                                            animated=False,
                                                            color="info",
                                                            style={"height": "26px", "marginTop": "8px"},
                                                        ),
                                                    ],
                                                    className="mt-2",
                                                ),
                                                html.Hr(),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "Outputs and CSV download appear as soon as the fetch finishes.",
                                                            style={"color": "#5f6368", "fontSize": "14px"},
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3 shadow-sm",
                                    style={"border": "1px solid #e0e0e0"},
                                ),
                                # Wind turbine settings (visible when wind selected)
                                html.Div(
                                    id="wind-settings-container",
                                    children=[
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        "Advanced turbine settings ",
                                                                        html.Span(
                                                                            html.I(className="bi bi-info-circle-fill"),
                                                                            id="power-curve-info",
                                                                            style={
                                                                                "cursor": "pointer",
                                                                                "color": google_colors["blue"],
                                                                                "fontSize": "14px",
                                                                            },
                                                                        ),
                                                                        dbc.Popover(
                                                                            [
                                                                                dbc.PopoverHeader("Wind Power Curve Formula"),
                                                                                dbc.PopoverBody(
                                                                                    [
                                                                                        html.P([
                                                                                            html.Strong("Below cut-in speed:"), " Power = 0"
                                                                                        ]),
                                                                                        html.P([
                                                                                            html.Strong("Ramp region (cut-in to rated):"),
                                                                                            html.Br(),
                                                                                            "Power = Rated MW × ((wind - cut_in) / (rated - cut_in))³"
                                                                                        ]),
                                                                                        html.P([
                                                                                            html.Strong("Rated region (rated to cut-out):"),
                                                                                            html.Br(),
                                                                                            "Power = Rated MW × Max CF"
                                                                                        ]),
                                                                                        html.P([
                                                                                            html.Strong("Above cut-out:"), " Power = 0 (safety shutdown)"
                                                                                        ]),
                                                                                        html.Hr(),
                                                                                        html.P([
                                                                                            html.Strong("Max Plant CF"), " accounts for wake losses, ",
                                                                                            "availability (~97%), and spatial diversity. ",
                                                                                            "Real wind farms rarely exceed 85-95% of nameplate."
                                                                                        ], style={"fontSize": "12px", "color": "#5f6368"}),
                                                                                        html.Hr(),
                                                                                        html.P([
                                                                                            html.Strong("Hub Height Extrapolation"),
                                                                                            html.Br(),
                                                                                            "Wind speed at hub height = ws_100m × (hub_h / 100)^α",
                                                                                            html.Br(),
                                                                                            "Uses power law to adjust ERA5 100m wind to your turbine's hub height."
                                                                                        ], style={"fontSize": "12px", "color": "#5f6368"}),
                                                                                    ]
                                                                                ),
                                                                            ],
                                                                            target="power-curve-info",
                                                                            trigger="click",
                                                                            placement="right",
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "fontWeight": "700",
                                                                        "color": google_colors["red"],
                                                                    },
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "Show/Hide",
                                                                    id="advanced-toggle",
                                                                    color="light",
                                                                    style={"fontWeight": "600"},
                                                                    className="float-end",
                                                                ),
                                                                width=3,
                                                            ),
                                                        ],
                                                        align="center",
                                                    ),
                                                    style={"backgroundColor": "#fff"},
                                                ),
                                                dbc.Collapse(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Cut-in (m/s)"),
                                                                            dbc.Input(
                                                                                id="cut-in",
                                                                                type="number",
                                                                                value=3.0,
                                                                                step=0.5,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Rated speed (m/s)"),
                                                                            dbc.Input(
                                                                                id="rated-speed",
                                                                                type="number",
                                                                                value=12.0,
                                                                                step=0.5,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Cut-out (m/s)"),
                                                                            dbc.Input(
                                                                                id="cut-out",
                                                                                type="number",
                                                                                value=25.0,
                                                                                step=0.5,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Turbine rating (MW)"),
                                                                            dbc.Input(
                                                                                id="turbine-mw",
                                                                                type="number",
                                                                                value=4.0,
                                                                                step=0.1,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Max plant CF (%)"),
                                                                            dbc.Input(
                                                                                id="max-plant-cf",
                                                                                type="number",
                                                                                value=90,
                                                                                min=50,
                                                                                max=100,
                                                                                step=1,
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Plant CF notes"),
                                                                            html.Div(
                                                                                "Real wind farms rarely exceed 85-95% of nameplate due to wake losses, availability, and grid constraints.",
                                                                                style={"color": "#5f6368", "fontSize": "13px"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Hub height (m)"),
                                                                            dbc.Input(
                                                                                id="hub-height",
                                                                                type="number",
                                                                                value=100,
                                                                                min=50,
                                                                                max=200,
                                                                                step=5,
                                                                            ),
                                                                        ],
                                                                        width=2,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Shear source"),
                                                                            dbc.Switch(
                                                                                id="use-measured-shear",
                                                                                label="Use ERA5 measured",
                                                                                value=True,
                                                                                style={"marginTop": "5px"},
                                                                            ),
                                                                        ],
                                                                        width=2,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Manual shear (if not measured)"),
                                                                            dbc.Input(
                                                                                id="shear-exponent",
                                                                                type="number",
                                                                                value=0.14,
                                                                                min=0.05,
                                                                                max=0.35,
                                                                                step=0.01,
                                                                                disabled=True,
                                                                            ),
                                                                        ],
                                                                        width=3,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Shear guide"),
                                                                            html.Div(
                                                                                "0.10 water | 0.14 open | 0.20 suburban | 0.25+ forest",
                                                                                style={"color": "#5f6368", "fontSize": "12px"},
                                                                            ),
                                                                        ],
                                                                        width=5,
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Switch(
                                                                                id="apply-density-correction",
                                                                                label="Apply air density correction",
                                                                                value=True,
                                                                                style={"marginTop": "5px"},
                                                                            ),
                                                                        ],
                                                                        width=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            html.Div(
                                                                                "Adjusts power for actual air density vs standard (1.225 kg/m³). "
                                                                                "Lower density at high altitude/temperature reduces power ~10-20%.",
                                                                                style={"color": "#5f6368", "fontSize": "12px"},
                                                                            ),
                                                                        ],
                                                                        width=8,
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                    id="advanced-collapse",
                                                    is_open=True,
                                                ),
                                            ],
                                            className="shadow-sm",
                                            style={"border": "1px solid #e0e0e0"},
                                        ),
                                    ],
                                ),
                                # Solar settings (visible when solar selected)
                                html.Div(
                                    id="solar-settings-container",
                                    style={"display": "none"},
                                    children=[
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    "Solar array settings",
                                                                    style={
                                                                        "fontWeight": "700",
                                                                        "color": google_colors["yellow"],
                                                                    },
                                                                )
                                                            ),
                                                            dbc.Col(
                                                                dbc.Button(
                                                                    "Show/Hide",
                                                                    id="solar-advanced-toggle",
                                                                    color="light",
                                                                    style={"fontWeight": "600"},
                                                                    className="float-end",
                                                                ),
                                                                width=3,
                                                            ),
                                                        ],
                                                        align="center",
                                                    ),
                                                    style={"backgroundColor": "#fff"},
                                                ),
                                                dbc.Collapse(
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Panel tilt (degrees)"),
                                                                            dbc.Input(
                                                                                id="panel-tilt",
                                                                                type="number",
                                                                                value=30.0,
                                                                                step=1,
                                                                                min=0,
                                                                                max=90,
                                                                            ),
                                                                            html.Small(
                                                                                "0° = flat, typical: latitude angle",
                                                                                style={"color": "#5f6368"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Panel azimuth (degrees)"),
                                                                            dbc.Input(
                                                                                id="panel-azimuth",
                                                                                type="number",
                                                                                value=180.0,
                                                                                step=5,
                                                                                min=0,
                                                                                max=360,
                                                                            ),
                                                                            html.Small(
                                                                                "180° = south (N. hemisphere)",
                                                                                style={"color": "#5f6368"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Inverter Load Ratio"),
                                                                            dbc.Input(
                                                                                id="ilr",
                                                                                type="number",
                                                                                value=1.25,
                                                                                step=0.05,
                                                                                min=1.0,
                                                                                max=2.0,
                                                                            ),
                                                                            html.Small(
                                                                                "DC/AC ratio (typ. 1.2-1.4)",
                                                                                style={"color": "#5f6368"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Hr(),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("System efficiency (%)"),
                                                                            dbc.Input(
                                                                                id="system-efficiency",
                                                                                type="number",
                                                                                value=86,
                                                                                step=1,
                                                                                min=50,
                                                                                max=100,
                                                                            ),
                                                                            html.Small(
                                                                                "Soiling, wiring, temp losses",
                                                                                style={"color": "#5f6368"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("DC capacity (kW, auto from peak hour)"),
                                                                            dbc.Input(
                                                                                id="dc-capacity",
                                                                                type="number",
                                                                                value=None,
                                                                                step=100,
                                                                                min=0,
                                                                                placeholder="Computed after fetch",
                                                                                disabled=True,
                                                                            ),
                                                                            html.Small(
                                                                                "We derive DC capacity from the peak hour of AC output (AC * ILR).",
                                                                                style={"color": "#5f6368"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Notes"),
                                                                            html.Div(
                                                                                "ILR clips DC output at inverter capacity. Higher ILR = more clipping but better low-light performance.",
                                                                                style={"color": "#5f6368", "fontSize": "13px"},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                    id="solar-advanced-collapse",
                                                    is_open=True,
                                                ),
                                            ],
                                            className="shadow-sm",
                                            style={"border": "1px solid #e0e0e0"},
                                        ),
                                    ],
                                ),
                            ],
                            md=7,
                        ),
                    ],
                    className="g-3",
                ),
                html.Div(
                    id="outputs-container",
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "Outputs",
                                                            style={"fontWeight": "700", "color": google_colors["blue"]},
                                                        ),
                                                        # Year view selector for multi-year mode
                                                        html.Div(
                                                            [
                                                                html.Span(" - View: ", style={"fontWeight": "400", "color": "#5f6368", "marginRight": "4px"}),
                                                                html.Div(
                                                                    dcc.Dropdown(
                                                                        id="view-year-dropdown",
                                                                        options=[],
                                                                        value=None,
                                                                        clearable=False,
                                                                        disabled=True,
                                                                        style={"width": "90px"},
                                                                    ),
                                                                    style={"display": "inline-block", "minWidth": "90px"},
                                                                ),
                                                            ],
                                                            id="view-year-container",
                                                            style={"display": "none", "alignItems": "center"},
                                                        ),
                                                    ],
                                                    style={"display": "flex", "alignItems": "center", "gap": "4px"},
                                                )
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "Download hourly CSV",
                                                    id="download-btn",
                                                    color="success",
                                                    style={
                                                        "backgroundColor": google_colors["green"],
                                                        "borderColor": google_colors["green"],
                                                        "fontWeight": "600",
                                                    },
                                                    className="float-end",
                                                    disabled=True,
                                                ),
                                                width=4,
                                            ),
                                        ],
                                        align="center",
                                    ),
                                    style={"backgroundColor": "#fff"},
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="summary-text",
                                            style={"marginBottom": "12px", "color": "#3c4043", "fontWeight": "500"},
                                        ),
                                        dcc.Graph(id="shape-graph", style={"height": "360px"}),
                                        html.Div(id="stats-cards", style={"marginTop": "8px"}),
                                        # Curtailment analysis section
                                        html.Div(
                                            id="curtailment-section",
                                            children=[
                                                html.Hr(style={"margin": "24px 0 16px 0"}),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "Curtailment Analysis",
                                                            style={"fontWeight": "600", "color": "#202124", "fontSize": "16px"},
                                                        ),
                                                        html.Span(
                                                            " - Explore how curtailing peak output affects hourly matching",
                                                            style={"color": "#5f6368", "fontSize": "13px", "marginLeft": "8px"},
                                                        ),
                                                    ],
                                                    style={"marginBottom": "16px"},
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dcc.Graph(
                                                                    id="curtailment-curve-graph",
                                                                    style={"height": "300px"},
                                                                ),
                                                            ],
                                                            md=8,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Cap Level (% of peak)",
                                                                            style={"fontWeight": "500", "color": "#5f6368", "marginBottom": "8px"},
                                                                        ),
                                                                        dcc.Slider(
                                                                            id="curtailment-slider",
                                                                            min=0,
                                                                            max=100,
                                                                            step=1,
                                                                            value=100,  # Start at 100% cap (no curtailment)
                                                                            marks={i: f"{i}%" for i in range(0, 101, 20)},
                                                                            tooltip={"placement": "bottom", "always_visible": True},
                                                                        ),
                                                                    ],
                                                                    style={"marginBottom": "20px"},
                                                                ),
                                                                html.Div(id="curtailment-stats", style={"marginTop": "12px"}),
                                                            ],
                                                            md=4,
                                                            style={"display": "flex", "flexDirection": "column", "justifyContent": "center"},
                                                        ),
                                                    ],
                                                    className="g-3",
                                                ),
                                            ],
                                            style={"display": "none"},  # Hidden until data is available
                                        ),
                                        dcc.Download(id="download-data"),
                                    ]
                                ),
                            ],
                            className="mt-4 shadow-sm",
                            style={"border": "1px solid #e0e0e0"},
                        )
                    ],
                    style={"display": "none"},  # hidden until data arrives
                ),
                dcc.Interval(id="progress-interval", interval=1000, n_intervals=0, disabled=True),
                dcc.Store(id="location-store", data={"lat": default_lat, "lon": default_lon}),
                dcc.Store(id="year-store", data=2024),
                dcc.Store(id="energy-type-store", data="wind"),
                dcc.Store(id="energy-data-store"),
                # Multi-year mode stores
                dcc.Store(id="multi-year-mode-store", data=False),
                dcc.Store(id="year-range-store", data={"start": 2024, "end": 2024}),
                dcc.Store(id="multi-year-data-store", data={}),
                dcc.Store(id="fetched-years-store", data=[]),
            ],
            fluid=True,
            style={"padding": "22px"},
        )
    ],
    style={
        "background": "linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%)",
        "minHeight": "100vh",
        "fontFamily": "'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif",
    },
)


# ---- Callbacks ----


@app.callback(
    Output("location-store", "data"),
    Output("lat-input", "value"),
    Output("lon-input", "value"),
    Input("location-marker", "data"),
    State("location-store", "data"),
    prevent_initial_call=True,
)
def sync_from_marker_drag(marker_data, stored_loc):
    """Update store and inputs when marker is dragged (via eventHandlers)."""
    if marker_data is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Parse the latlng from dragend event
    try:
        lat = float(marker_data.get("lat"))
        lon = float(marker_data.get("lng"))
    except (ValueError, TypeError, AttributeError):
        return dash.no_update, dash.no_update, dash.no_update
    
    return {"lat": lat, "lon": lon}, round(lat, 4), round(lon, 4)


@app.callback(
    Output("location-marker", "position"),
    Output("location-store", "data", allow_duplicate=True),
    Input("location-map", "click_lat_lng"),
    Input("lat-input", "value"),
    Input("lon-input", "value"),
    State("location-store", "data"),
    prevent_initial_call=True,
)
def sync_marker_from_inputs(map_click, lat_val, lon_val, stored_loc):
    """Update marker position from map click or manual input."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered_id
    old_lat = stored_loc.get("lat", default_lat)
    old_lon = stored_loc.get("lon", default_lon)

    if trigger_id == "location-map" and map_click:
        lat, lon = map_click
    elif trigger_id in ("lat-input", "lon-input"):
        lat = lat_val if lat_val is not None else old_lat
        lon = lon_val if lon_val is not None else old_lon
    else:
        return dash.no_update, dash.no_update

    return [lat, lon], {"lat": lat, "lon": lon}


@app.callback(
    Output("location-map", "center"),
    Input("location-store", "data"),
)
def update_map_center(loc):
    lat, lon = loc["lat"], loc["lon"]
    return [lat, lon]


@app.callback(
    Output("target-mwh-display", "children"),
    Input("target-mwh", "value"),
)
def show_target_mwh(val):
    """Reflect slider value next to the control."""
    if val is None:
        return "Choose an annual GWh purchase"
    return f"{val:,.2f} GWh per year"




@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-toggle", "n_clicks"),
    State("advanced-collapse", "is_open"),
)
def toggle_advanced(n, is_open):
    if not n:
        return is_open
    return not is_open


@app.callback(
    Output("solar-advanced-collapse", "is_open"),
    Input("solar-advanced-toggle", "n_clicks"),
    State("solar-advanced-collapse", "is_open"),
)
def toggle_solar_advanced(n, is_open):
    if not n:
        return is_open
    return not is_open


@app.callback(
    Output("shear-exponent", "disabled"),
    Input("use-measured-shear", "value"),
)
def toggle_shear_input(use_measured):
    """Disable manual shear input when using measured ERA5 shear."""
    return use_measured


@app.callback(
    Output("outputs-container", "style"),
    Output("download-btn", "disabled"),
    Input("energy-data-store", "data"),
    Input("progress-interval", "n_intervals"),
)
def toggle_outputs_visibility(data, _n):
    """Hide outputs until a fetch has produced data."""
    state = get_fetch_state()
    has_data = data is not None or (state.get("result") is not None or state.get("last_result") is not None)
    style = {} if has_data else {"display": "none"}
    return style, not has_data


# ---- Multi-year mode callbacks ----


@app.callback(
    Output("single-year-container", "style"),
    Output("year-range-container", "style"),
    Output("multi-year-mode-store", "data"),
    Input("multi-year-toggle", "value"),
)
def toggle_multi_year_mode(is_multi_year):
    """Show/hide single year vs range picker based on toggle."""
    if is_multi_year:
        return {"display": "none"}, {"display": "block"}, True
    else:
        return {"display": "block"}, {"display": "none"}, False


@app.callback(
    Output("year-range-store", "data"),
    Output("year-range-validation", "children"),
    Input("start-year-dropdown", "value"),
    Input("end-year-dropdown", "value"),
)
def validate_year_range(start_year, end_year):
    """Validate year range and update store."""
    if start_year is None or end_year is None:
        return dash.no_update, ""

    if start_year > end_year:
        return {"start": start_year, "end": end_year}, "Start must be <= End"

    num_years = end_year - start_year + 1
    if num_years > 10:
        return {"start": start_year, "end": end_year}, f"Max 10 years ({num_years} selected)"

    return {"start": start_year, "end": end_year}, ""


@app.callback(
    Output("energy-data-store", "data", allow_duplicate=True),
    Output("year-store", "data", allow_duplicate=True),
    Input("view-year-dropdown", "value"),
    State("multi-year-data-store", "data"),
    State("multi-year-mode-store", "data"),
    prevent_initial_call=True,
)
def switch_view_year(selected_year, multi_year_data, is_multi_year):
    """Switch displayed year when user selects a different year in multi-year mode."""
    print(f"[switch_view_year] Called with year={selected_year}, is_multi_year={is_multi_year}, has_data={bool(multi_year_data)}")

    if selected_year is None:
        return dash.no_update, dash.no_update

    if not multi_year_data:
        print("[switch_view_year] No multi_year_data available")
        return dash.no_update, dash.no_update

    # Try multiple key formats (JSON may stringify keys, and value could be int or str)
    year_payload = None
    keys_tried = []
    for key_variant in [selected_year, str(selected_year), int(selected_year) if str(selected_year).lstrip('-').isdigit() else None]:
        if key_variant is not None:
            keys_tried.append(key_variant)
            if key_variant in multi_year_data:
                year_payload = multi_year_data[key_variant]
                print(f"[switch_view_year] Found data for year {selected_year} using key {key_variant}")
                break

    if not year_payload:
        print(f"[switch_view_year] Could not find year {selected_year}. Tried keys: {keys_tried}. Available: {list(multi_year_data.keys())}")
        return dash.no_update, dash.no_update

    return year_payload, selected_year


@app.callback(
    Output("main-title", "children"),
    Output("main-description", "children"),
    Output("location-card-header", "children"),
    Output("wind-settings-container", "style"),
    Output("solar-settings-container", "style"),
    Output("energy-type-store", "data"),
    Input("energy-type-dropdown", "value"),
)
def update_energy_type_ui(energy_type):
    """Update UI elements based on selected energy type."""
    if energy_type == "solar":
        return (
            "Forecast-quality shapes for your solar farm",
            "Drop a pin for location, choose how many RECs you purchase, tweak solar array settings, then fetch. Outputs and CSV download unlock as soon as the run finishes.",
            "Where is your solar farm?",
            {"display": "none"},
            {"display": "block"},
            "solar",
        )
    else:
        return (
            "Forecast-quality shapes for your wind farm",
            "Drop a pin for location, choose how many RECs you purchase, tweak advanced turbine settings, then fetch. Outputs and CSV download unlock as soon as the run finishes.",
            "Where is your wind farm?",
            {"display": "block"},
            {"display": "none"},
            "wind",
        )


@app.callback(
    Output("cached-markers", "children"),
    Input("energy-data-store", "data"),
    Input("location-store", "data"),
    Input("year-dropdown", "value"),
    Input("energy-type-store", "data"),
)
def update_cached_markers(_, loc, year, energy_type):
    """Show green circle markers for all cached locations."""
    cached = get_cached_locations(year=year, energy_type=energy_type or "wind")
    current_lat = round(loc.get("lat", default_lat), 2)
    current_lon = round(loc.get("lon", default_lon), 2)
    
    markers = []
    for lat, lon in cached:
        # Check if this cached location matches current selection
        is_current = (lat == current_lat and lon == current_lon)
        markers.append(
            dl.CircleMarker(
                center=[lat, lon],
                radius=8 if is_current else 6,
                color=google_colors["green"],
                fill=True,
                fillColor=google_colors["green"],
                fillOpacity=0.7 if is_current else 0.5,
                children=dl.Tooltip(f"Cached: ({lat}, {lon})"),
            )
        )
    return markers


@app.callback(
    Output("status-text", "children", allow_duplicate=True),
    Output("year-store", "data"),
    Output("energy-data-store", "data", allow_duplicate=True),
    Output("multi-year-data-store", "data", allow_duplicate=True),
    Output("fetched-years-store", "data", allow_duplicate=True),
    Input("fetch-btn", "n_clicks"),
    State("location-store", "data"),
    State("year-dropdown", "value"),
    State("energy-type-store", "data"),
    State("multi-year-mode-store", "data"),
    State("year-range-store", "data"),
    prevent_initial_call=True,
)
def start_fetch(n, loc, year, energy_type, multi_year_mode, year_range):
    if not n:
        return (dash.no_update,) * 5
    state = get_fetch_state()
    if state["status"] == "running":
        return ("Fetch already in progress...",) + (dash.no_update,) * 4
    energy_type = energy_type or "wind"
    type_label = "solar" if energy_type == "solar" else "wind"

    if multi_year_mode:
        # Multi-year fetch
        start_year = year_range.get("start", 2024)
        end_year = year_range.get("end", 2024)
        years = list(range(start_year, end_year + 1))

        # Validate
        if len(years) > 10:
            return ("Error: Maximum 10 years allowed",) + (dash.no_update,) * 4
        if start_year > end_year:
            return ("Error: Start year must be <= end year",) + (dash.no_update,) * 4

        set_fetch_state(
            status="running",
            progress=0,
            total=len(years),
            message=f"Starting multi-year fetch ({len(years)} years)...",
            result=None,
            last_result=None,
            error=None,
            multi_year_results={},
            years_completed=[],
            years_pending=list(years),
            current_year=years[0],
        )

        thread = threading.Thread(
            target=fetch_multi_year_worker,
            args=(loc["lat"], loc["lon"], years, energy_type),
            daemon=True,
        )
        thread.start()

        return (
            f"Started {type_label} fetch for {start_year}-{end_year} ({len(years)} years)...",
            end_year,  # Default to most recent year for visualization
            None,  # Clear energy-data-store
            {},    # Clear multi-year-data-store
            [],    # Clear fetched-years-store
        )
    else:
        # Single year fetch (existing behavior)
        set_fetch_state(
            status="running",
            progress=0,
            total=1,
            message="Starting...",
            result=None,
            last_result=None,
            error=None,
        )
        thread = threading.Thread(
            target=fetch_worker,
            args=(loc["lat"], loc["lon"], year, energy_type),
            daemon=True
        )
        thread.start()
        # Clear prior payload so outputs blank until the new fetch completes
        return (
            f"Started {type_label} fetch for {year} (about a minute). Progress will update below.",
            year,
            None,
            {},   # Clear multi-year-data-store
            [],   # Clear fetched-years-store
        )


@app.callback(
    Output("progress-bar", "value"),
    Output("progress-bar", "label"),
    Output("progress-bar", "color"),
    Output("progress-bar", "striped"),
    Output("progress-bar", "animated"),
    Output("status-text", "children", allow_duplicate=True),
    Output("energy-data-store", "data"),
    Output("multi-year-data-store", "data"),
    Output("fetched-years-store", "data"),
    Output("view-year-dropdown", "options"),
    Output("view-year-dropdown", "value"),
    Output("view-year-dropdown", "disabled"),
    Output("view-year-container", "style"),
    Input("progress-interval", "n_intervals"),
    State("multi-year-mode-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def poll_progress(_, is_multi_year):
    state = get_fetch_state()
    status = state["status"]
    val = 0
    label = "Idle"
    color = "success"
    striped = False
    animated = False
    status_text = state.get("message", "")
    data_out = dash.no_update
    multi_data_out = dash.no_update
    fetched_years_out = dash.no_update
    view_options = dash.no_update
    view_value = dash.no_update
    view_disabled = dash.no_update
    view_container_style = dash.no_update

    if status == "running":
        total = max(state.get("total", 1), 1)
        progress = min(state.get("progress", 0), total)
        val = int((progress / total) * 100)
        label = f"{progress}/{total}"
        color = "info"
        striped = True
        animated = True
    elif status == "done":
        val = 100
        label = "Done"
        color = "success"
        status_text = state.get("message", "Done")

        result = state.get("result")

        if is_multi_year and isinstance(result, dict) and len(result) > 0:
            # Multi-year fetch completed
            multi_results = state.get("multi_year_results", result)
            years_completed = sorted(multi_results.keys(), reverse=True)

            # Build dropdown options
            view_options = [{"label": str(y), "value": y} for y in years_completed]
            default_year = years_completed[0] if years_completed else None
            view_value = default_year
            view_disabled = False
            view_container_style = {"display": "flex", "alignItems": "center"}

            # Get the most recent year's data for initial display
            data_out = multi_results.get(default_year) if default_year else None
            multi_data_out = multi_results
            fetched_years_out = years_completed

            # Clear stored result after emitting
            set_fetch_state(
                status="done",
                progress=state.get("progress", 1),
                total=state.get("total", 1),
                message=status_text,
                result=None,
                last_result=multi_results,
                error=None,
                duration=state.get("duration", 0.0),
                multi_year_results={},
            )
        else:
            # Single year fetch completed
            data_out = result
            view_container_style = {"display": "none"}

            # Clear stored result after emitting once so new sessions don't auto-load
            if data_out is not None:
                set_fetch_state(
                    status="done",
                    progress=state.get("progress", 1),
                    total=state.get("total", 1),
                    message=status_text,
                    result=None,
                    last_result=data_out,
                    error=None,
                    duration=state.get("duration", 0.0),
                )
    elif status == "error":
        val = 0
        label = "Error"
        color = "danger"
        status_text = f"Error: {state.get('error')}"

    return (
        val, label, color, striped, animated, status_text, data_out,
        multi_data_out, fetched_years_out, view_options, view_value,
        view_disabled, view_container_style
    )


@app.callback(
    Output("progress-interval", "disabled"),
    Input("fetch-btn", "n_clicks"),
    Input("progress-interval", "n_intervals"),
)
def toggle_progress_interval(n_clicks, _n):
    state = get_fetch_state()
    status = state.get("status", "idle")
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    if triggered_id == "fetch-btn" and n_clicks:
        return False
    return status != "running"


def max_consecutive_true(mask):
    max_run = 0
    current = 0
    for is_true in mask:
        if is_true:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def estimate_weibull_params(samples):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    samples = samples[samples > 0]
    if samples.size == 0:
        return None, None
    mean = samples.mean()
    std = samples.std()
    if mean <= 0 or std <= 0:
        return None, None
    k = (std / mean) ** -1.086
    lam = mean / math.gamma(1.0 + 1.0 / k)
    return k, lam


def calculate_curtailment_curve(cf_series, num_points=101):
    """
    Calculate curtailment curve: as you lower the production cap, matching % increases.

    Args:
        cf_series: Array of capacity factors (0-1 range)
        num_points: Number of points on the curve (0% to 100% curtailment)

    Returns:
        dict with:
          - curtailment_pct: array from 0 to 100 (curtailment %)
          - cap_pct: array from 100 to 0 (production cap as % of nameplate)
          - matching_pct: array of matching % at each curtailment level
          - energy_delivered_pct: array of energy delivered as % of uncurtailed
    """
    cf = np.asarray(cf_series, dtype=float)
    cf = cf[np.isfinite(cf)]
    if cf.size == 0:
        return None

    # Capacity factor of uncurtailed generation (baseline)
    baseline_cf = np.mean(cf)
    total_hours = len(cf)

    curtailment_levels = np.linspace(0, 100, num_points)  # 0% to 100%
    cap_levels = 100 - curtailment_levels  # 100% down to 0%
    matching_pcts = []
    energy_delivered_pcts = []

    for cap_pct in cap_levels:
        # Cap is the maximum production level as fraction of nameplate
        cap_fraction = cap_pct / 100.0

        # Curtailed CF: clip generation at the cap
        curtailed_cf = np.minimum(cf, cap_fraction)

        # Energy delivered as fraction of uncurtailed energy
        if baseline_cf > 0:
            energy_delivered = np.mean(curtailed_cf) / baseline_cf
        else:
            energy_delivered = 0
        energy_delivered_pcts.append(energy_delivered * 100)

        # Matching %: what fraction of the "box" (cap * hours) is filled?
        # The "box" represents baseload-equivalent at the cap level
        if cap_fraction > 0:
            box_area = cap_fraction * total_hours  # Total possible at this cap
            actual_area = np.sum(curtailed_cf)  # Actual generation (sum of CFs)
            matching = actual_area / box_area
        else:
            matching = 100.0  # At 0% cap, trivially 100% matching
        matching_pcts.append(matching * 100)

    return {
        "curtailment_pct": curtailment_levels,
        "cap_pct": cap_levels,
        "matching_pct": np.array(matching_pcts),
        "energy_delivered_pct": np.array(energy_delivered_pcts),
        "baseline_cf": baseline_cf,
    }


def build_stat_card(title, value, subtitle=None):
    body = [
        html.Div(title, style={"fontSize": "13px", "color": "#5f6368", "textTransform": "uppercase", "letterSpacing": "0.4px"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": "#202124", "marginTop": "6px"}),
    ]
    if subtitle:
        body.append(html.Div(subtitle, style={"fontSize": "12px", "color": "#5f6368", "marginTop": "4px"}))
    return dbc.Card(
        dbc.CardBody(body),
        className="shadow-sm h-100",
        style={"border": "1px solid #e0e0e0"},
    )


def calculate_multi_year_shape_stats(multi_year_data, energy_type="wind"):
    """
    Calculate shape quality statistics across multiple years.

    Since volume is fixed (user purchases a fixed GWh), shape quality matters:
    - Weibull k (shape) distribution: higher k = less variability, smoother output
    - Year-to-year resource variability
    """
    from io import StringIO

    if not multi_year_data:
        return {}

    weibull_k_values = []
    weibull_lambda_values = []

    # Handle both int and string keys from JSON serialization
    items = [(int(k) if str(k).isdigit() else k, v) for k, v in multi_year_data.items()]
    for year, payload in sorted(items, key=lambda x: x[0]):
        df_json = payload.get("data") if isinstance(payload, dict) else payload
        if not df_json:
            continue

        df = pd.read_json(StringIO(df_json), orient="split")
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Calculate Weibull for wind speed (wind only)
        if energy_type == "wind" and "wind_speed_100m" in df.columns:
            ws = df["wind_speed_100m"].values
            k, lam = estimate_weibull_params(ws)
            if k is not None:
                weibull_k_values.append({"year": year, "k": k, "lambda": lam})
                weibull_lambda_values.append(lam)

    # Aggregate statistics
    stats = {
        "years_analyzed": len(weibull_k_values) if energy_type == "wind" else len(multi_year_data),
        "weibull_k_values": weibull_k_values,
    }

    if weibull_k_values:
        k_vals = [w["k"] for w in weibull_k_values]
        stats["weibull_k_p10"] = float(np.percentile(k_vals, 10))
        stats["weibull_k_p50"] = float(np.percentile(k_vals, 50))
        stats["weibull_k_p90"] = float(np.percentile(k_vals, 90))
        stats["weibull_k_mean"] = float(np.mean(k_vals))
        stats["weibull_k_std"] = float(np.std(k_vals))

    return stats


def calculate_annual_cf_stats(multi_year_data, compute_cf_func):
    """
    Calculate capacity factor statistics across years.

    Args:
        multi_year_data: Dict of {year: payload}
        compute_cf_func: Function that takes a DataFrame and returns CF series
    """
    from io import StringIO

    annual_cfs = []

    # Handle both int and string keys from JSON serialization
    items = [(int(k) if str(k).isdigit() else k, v) for k, v in multi_year_data.items()]
    for year, payload in sorted(items, key=lambda x: x[0]):
        df_json = payload.get("data") if isinstance(payload, dict) else payload
        if not df_json:
            continue

        df = pd.read_json(StringIO(df_json), orient="split")
        df["datetime"] = pd.to_datetime(df["datetime"])

        try:
            cf_series = compute_cf_func(df)
            if cf_series is not None and len(cf_series) > 0:
                annual_cfs.append({
                    "year": year,
                    "mean_cf": float(cf_series.mean()),
                    "std_cf": float(cf_series.std()),
                    "cv_cf": float(cf_series.std() / cf_series.mean()) if cf_series.mean() > 0 else 0,
                })
        except Exception:
            continue

    if not annual_cfs:
        return {}

    mean_cfs = [a["mean_cf"] for a in annual_cfs]

    return {
        "annual_cf_data": annual_cfs,
        "inter_annual_cf_mean": float(np.mean(mean_cfs)),
        "inter_annual_cf_std": float(np.std(mean_cfs)),
        "inter_annual_cf_cv": float(np.std(mean_cfs) / np.mean(mean_cfs)) if np.mean(mean_cfs) > 0 else 0,
        "cf_p10": float(np.percentile(mean_cfs, 10)),
        "cf_p50": float(np.percentile(mean_cfs, 50)),
        "cf_p90": float(np.percentile(mean_cfs, 90)),
    }


def build_multi_year_stats_section(shape_stats, cf_stats, energy_type="wind"):
    """Build UI cards for multi-year shape statistics."""
    cards = []

    if energy_type == "wind" and shape_stats.get("weibull_k_values"):
        # Weibull k distribution card (P10/P50/P90)
        k_p10 = shape_stats.get("weibull_k_p10", 0)
        k_p50 = shape_stats.get("weibull_k_p50", 0)
        k_p90 = shape_stats.get("weibull_k_p90", 0)

        cards.append(
            dbc.Col(
                build_stat_card(
                    "Weibull k (P10/P50/P90)",
                    f"{k_p10:.2f} / {k_p50:.2f} / {k_p90:.2f}",
                    "Shape parameter across years (higher = smoother)"
                ),
                md=4, sm=6, xs=12, className="mb-3"
            )
        )

        # Year-over-year k variation
        k_std = shape_stats.get("weibull_k_std", 0)
        cards.append(
            dbc.Col(
                build_stat_card(
                    "Weibull k StdDev",
                    f"{k_std:.3f}",
                    "Year-to-year shape consistency"
                ),
                md=4, sm=6, xs=12, className="mb-3"
            )
        )

    if cf_stats:
        # Inter-annual CF variation
        cf_cv = cf_stats.get("inter_annual_cf_cv", 0)
        cf_std = cf_stats.get("inter_annual_cf_std", 0)

        cards.append(
            dbc.Col(
                build_stat_card(
                    "Annual CF Variation",
                    f"CV={cf_cv:.1%}, Std={cf_std:.3f}",
                    "Year-to-year resource variability"
                ),
                md=4, sm=6, xs=12, className="mb-3"
            )
        )

        # CF P10/P50/P90
        cf_p10 = cf_stats.get("cf_p10", 0)
        cf_p50 = cf_stats.get("cf_p50", 0)
        cf_p90 = cf_stats.get("cf_p90", 0)

        cards.append(
            dbc.Col(
                build_stat_card(
                    "Annual CF (P10/P50/P90)",
                    f"{cf_p10:.1%} / {cf_p50:.1%} / {cf_p90:.1%}",
                    "Exceedance probabilities across years"
                ),
                md=4, sm=6, xs=12, className="mb-3"
            )
        )

    if not cards:
        return html.Div()

    return html.Div([
        html.H6(
            "Multi-Year Shape Statistics",
            style={"fontWeight": "600", "marginTop": "16px", "marginBottom": "12px", "color": "#1a73e8"}
        ),
        dbc.Row(cards, className="g-3"),
    ])


@app.callback(
    Output("shape-graph", "figure"),
    Output("stats-cards", "children"),
    Output("summary-text", "children"),
    Input("energy-data-store", "data"),
    # Wind inputs
    Input("cut-in", "value"),
    Input("rated-speed", "value"),
    Input("cut-out", "value"),
    Input("turbine-mw", "value"),
    Input("max-plant-cf", "value"),
    Input("hub-height", "value"),
    Input("use-measured-shear", "value"),
    Input("shear-exponent", "value"),
    Input("apply-density-correction", "value"),
    # Solar inputs
    Input("panel-tilt", "value"),
    Input("panel-azimuth", "value"),
    Input("ilr", "value"),
    Input("system-efficiency", "value"),
    Input("dc-capacity", "value"),
    # Common
    Input("target-mwh", "value"),
    # Curtailment slider
    Input("curtailment-slider", "value"),
    State("location-store", "data"),
    State("year-store", "data"),
    State("multi-year-mode-store", "data"),
    State("multi-year-data-store", "data"),
    State("fetched-years-store", "data"),
    prevent_initial_call=True,
)
def update_analysis(json_data, cin, rated_spd, cout, turb_mw, max_plant_cf,
                    hub_height, use_measured_shear, shear_exponent, apply_density_correction,
                    panel_tilt, panel_azimuth, ilr, sys_eff, dc_cap,
                    target_mwh, cap_level_pct, loc, year, is_multi_year, multi_year_data, fetched_years):
    if json_data is None:
        # Fallback to the latest finished fetch in case the store didn't update
        state = get_fetch_state()
        if state.get("status") == "running":
            return dash.no_update, dash.no_update, dash.no_update
        json_data = state.get("result") or state.get("last_result")
        if json_data is None:
            return dash.no_update, dash.no_update, dash.no_update

    payload = json_data
    df_json = payload
    lat_source = None
    lon_source = None
    year_source = None
    energy_type = "wind"
    if isinstance(payload, dict):
        df_json = payload.get("data")
        lat_source = payload.get("lat")
        lon_source = payload.get("lon")
        year_source = payload.get("year")
        energy_type = payload.get("energy_type", "wind")

    if not df_json:
        return dash.no_update, dash.no_update, dash.no_update

    df = pd.read_json(StringIO(df_json), orient="split")
    df["datetime"] = pd.to_datetime(df["datetime"])

    lat = lat_source if lat_source is not None else loc.get("lat", default_lat)
    lon = lon_source if lon_source is not None else loc.get("lon", default_lon)
    year_val = year_source if year_source is not None else (year if year is not None else 2024)

    if energy_type == "solar":
        # Guard against None values for solar (dc_cap is computed automatically)
        if any(v is None for v in [panel_tilt, panel_azimuth, ilr, sys_eff, target_mwh]):
            return dash.no_update, dash.no_update, dash.no_update

        # Calculate solar power
        sys_eff_frac = sys_eff / 100.0
        # First pass with nominal DC to compute peak hour, then recompute using that peak
        nominal_dc_kw = 1000.0
        ac_power_kw_nominal, poa = apply_solar_power(
            df["ghi_wm2"], df["datetime"], lat, lon,
            panel_tilt, panel_azimuth, ilr, nominal_dc_kw, sys_eff_frac
        )
        peak_ac_kw = float(np.max(ac_power_kw_nominal)) if len(ac_power_kw_nominal) else 0.0
        dc_cap_auto_kw = peak_ac_kw * ilr if peak_ac_kw > 0 else nominal_dc_kw

        ac_power_kw, poa = apply_solar_power(
            df["ghi_wm2"], df["datetime"], lat, lon,
            panel_tilt, panel_azimuth, ilr, dc_cap_auto_kw, sys_eff_frac
        )
        df["ac_power_kw"] = ac_power_kw
        df["poa_wm2"] = poa

        ac_capacity_kw = dc_cap_auto_kw / ilr
        df["cf"] = df["ac_power_kw"] / ac_capacity_kw

        avg_cf = df["cf"].mean()
        # Scale to target energy (convert GWh to MWh)
        target_mwh = target_mwh * 1000 if target_mwh is not None else 0
        installed_kw_ac = target_mwh * 1000 / (8760 * avg_cf) if avg_cf > 0 else 0
        installed_mw_ac = installed_kw_ac / 1000
        df["scaled_mw"] = df["cf"] * installed_mw_ac
        df["scaled_mwh"] = df["scaled_mw"]

        # Plot color for solar
        plot_color = google_colors["yellow"]
        title_prefix = "Solar"
        summary = (
            f"Avg CF: {avg_cf:.2%}. AC capacity to meet {target_mwh/1000:.2f} GWh/year: "
            f"{installed_mw_ac:.2f} MW (DC: {dc_cap_auto_kw/1000:.2f} MWdc at peak-hour-based sizing). "
            f"ILR: {ilr:.2f}. Rows: {len(df)}. "
            f"Location: ({lat:.2f}, {lon:.2f}) | Year: {year_val}."
        )
    else:
        # Wind processing
        if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf, hub_height, target_mwh]):
            return dash.no_update, dash.no_update, dash.no_update

        hub_h = hub_height or 100

        # Determine shear exponent: use measured from ERA5 or manual input
        if use_measured_shear and "shear_exponent" in df.columns:
            # Use per-hour measured shear from ERA5 (average for display)
            shear_values = df["shear_exponent"].values
            avg_shear = df["shear_exponent"].mean()
            shear_source = "measured"
        else:
            # Use manual shear exponent (constant for all hours)
            shear_values = shear_exponent or 0.14
            avg_shear = shear_values
            shear_source = "manual"

        # Extrapolate wind speed from 100m to hub height using power law
        df["wind_speed_hub"] = extrapolate_wind_speed(df["wind_speed_100m"].values, hub_h, shear_values)

        # Convert max_plant_cf from percentage to fraction
        max_cf_frac = (max_plant_cf or 90) / 100.0
        df["power_mw_raw"] = apply_power_curve(
            df["wind_speed_hub"].values, cin, rated_spd, cout, turb_mw, max_cf_frac
        )

        # Apply air density correction if enabled
        if apply_density_correction and "air_density" in df.columns:
            df["power_mw"] = adjust_power_for_density(df["power_mw_raw"].values, df["air_density"].values)
            avg_density = df["air_density"].mean()
            density_note = f" Avg ρ={avg_density:.3f} kg/m³."
        else:
            df["power_mw"] = df["power_mw_raw"]
            density_note = ""

        df["cf"] = df["power_mw"] / turb_mw

        avg_cf = df["cf"].mean()
        target_mwh = target_mwh * 1000 if target_mwh is not None else 0
        installed_mw = target_mwh / (8760 * avg_cf) if avg_cf > 0 else 0
        df["scaled_mw"] = df["cf"] * installed_mw
        df["scaled_mwh"] = df["scaled_mw"]

        plot_color = google_colors["blue"]
        title_prefix = "Wind"
        summary = (
            f"Avg CF: {avg_cf:.2%}. Installed MW to meet {target_mwh/1000:.2f} GWh/year: "
            f"{installed_mw:.2f} MW. Hub: {hub_h}m, α={avg_shear:.2f} ({shear_source}).{density_note} "
            f"Location: ({lat:.2f}, {lon:.2f}) | Year: {year_val}."
        )

    # Google-style font settings
    google_font = "Roboto, Open Sans, Helvetica Neue, sans-serif"
    title_font = dict(family=google_font, size=16, color="#202124", weight=600)
    axis_font = dict(family=google_font, size=12, color="#5f6368")
    tick_font = dict(family=google_font, size=11, color="#5f6368")

    # Hourly shape plot (scaled)
    fig_shape = go.Figure()
    fig_shape.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["scaled_mw"],
            mode="lines",
            line=dict(color=plot_color, width=1.5),
            name="Scaled MW",
        )
    )

    # Add red dotted cap line based on cap level slider
    # cap_level_pct is 0-100 where 100 = peak (no curtailment), 0 = fully curtailed
    cap_pct = cap_level_pct if cap_level_pct is not None else 100
    peak_mw = df["scaled_mw"].max()
    cap_mw = peak_mw * (cap_pct / 100.0)

    if cap_pct < 100:  # Only show line if there's curtailment
        fig_shape.add_trace(
            go.Scatter(
                x=[df["datetime"].min(), df["datetime"].max()],
                y=[cap_mw, cap_mw],
                mode="lines",
                line=dict(color=google_colors["red"], width=2, dash="dot"),
                name=f"Cap: {cap_pct:.0f}% ({cap_mw:.1f} MW)",
                hoverinfo="name+y",
            )
        )

    fig_shape.update_layout(
        title=dict(text=f"{title_prefix} hourly shape (scaled to purchased GWh)", font=title_font),
        yaxis_title="Power (MW)",
        template="plotly_white",
        font=dict(family=google_font),
        yaxis=dict(title_font=axis_font, tickfont=tick_font, gridcolor="#e8eaed"),
        xaxis=dict(tickfont=tick_font, gridcolor="#e8eaed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=20),
        showlegend=True if cap_pct < 100 else False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    cf_series = df["cf"].replace([np.inf, -np.inf], np.nan).dropna()
    autocorr_1h = cf_series.autocorr(lag=1) if len(cf_series) > 1 else np.nan
    autocorr_24h = cf_series.autocorr(lag=24) if len(cf_series) > 24 else np.nan
    low_threshold = 0.1
    low_mask = (cf_series < low_threshold).values
    low_hours_pct = float(np.mean(low_mask)) * 100 if low_mask.size else 0.0
    max_low_streak = max_consecutive_true(low_mask) if low_mask.size else 0

    if energy_type == "solar":
        resource_series = df["ac_power_kw"].values if "ac_power_kw" in df.columns else df["cf"].values
        resource_unit = "kW"
        weibull_title = "Weibull fit"
        weibull_value = "N/A"
        weibull_sub = "Wind-only metric"
    else:
        resource_series = df["wind_speed_hub"].values
        resource_unit = "m/s"
        k, lam = estimate_weibull_params(resource_series)
        weibull_title = "Weibull fit"
        if k is None or lam is None:
            weibull_value = "Unavailable"
            weibull_sub = "Insufficient data"
        else:
            weibull_value = f"k={k:.2f}, lambda={lam:.2f}"
            weibull_sub = f"Fit on hub-height wind speed ({resource_unit})"

    ramp_series = df["power_mw"].values if "power_mw" in df.columns else df["scaled_mw"].values
    ramp_series = ramp_series[np.isfinite(ramp_series)]
    ramps = np.abs(np.diff(ramp_series)) if ramp_series.size > 1 else np.array([])
    ramp_p95 = float(np.nanpercentile(ramps, 95)) if ramps.size else 0.0

    single_year_cards = dbc.Row(
        [
            dbc.Col(build_stat_card("Autocorr (1h)", f"{autocorr_1h:.2f}" if np.isfinite(autocorr_1h) else "N/A",
                                    "Hour-to-hour persistence"), md=4, sm=6, xs=12, className="mb-3"),
            dbc.Col(build_stat_card("Autocorr (24h)", f"{autocorr_24h:.2f}" if np.isfinite(autocorr_24h) else "N/A",
                                    "Daily rhythm strength"), md=4, sm=6, xs=12, className="mb-3"),
            dbc.Col(build_stat_card("Low CF hours", f"{low_hours_pct:.1f}%",
                                    f"Share of hours below {low_threshold:.0%} CF"), md=4, sm=6, xs=12, className="mb-3"),
            dbc.Col(build_stat_card("Max low streak", f"{max_low_streak} hrs",
                                    f"Longest run below {low_threshold:.0%} CF"), md=4, sm=6, xs=12, className="mb-3"),
            dbc.Col(build_stat_card("P95 ramp", f"{ramp_p95:.2f} MW",
                                    "95th percentile hour-to-hour change"), md=4, sm=6, xs=12, className="mb-3"),
            dbc.Col(build_stat_card(weibull_title, weibull_value, weibull_sub),
                    md=4, sm=6, xs=12, className="mb-3"),
        ],
        className="g-3",
    )

    # Add multi-year statistics if applicable
    if is_multi_year and multi_year_data and len(multi_year_data) > 1:
        # Calculate multi-year shape statistics
        shape_stats = calculate_multi_year_shape_stats(multi_year_data, energy_type)

        # Create CF computation function based on energy type and current settings
        if energy_type == "wind":
            hub_h = hub_height or 100
            max_cf_frac = (max_plant_cf or 90) / 100.0

            def compute_cf_for_year(df_year):
                if use_measured_shear and "shear_exponent" in df_year.columns:
                    shear_vals = df_year["shear_exponent"].values
                else:
                    shear_vals = shear_exponent or 0.14
                ws_hub = extrapolate_wind_speed(df_year["wind_speed_100m"].values, hub_h, shear_vals)
                power = apply_power_curve(ws_hub, cin, rated_spd, cout, turb_mw, max_cf_frac)
                if apply_density_correction and "air_density" in df_year.columns:
                    power = adjust_power_for_density(power, df_year["air_density"].values)
                return pd.Series(power / turb_mw)
        else:
            # Solar CF computation
            tilt = panel_tilt or 30
            azim = panel_azimuth or 180
            dc_capacity_kw = dc_cap or 1000
            _ilr = ilr or 1.2
            ac_capacity_kw = dc_capacity_kw / _ilr
            efficiency = (sys_eff or 85) / 100.0

            def compute_cf_for_year(df_year):
                if "ghi_wm2" not in df_year.columns:
                    return pd.Series([])
                ghi = df_year["ghi_wm2"].values
                dt_series = df_year["datetime"]
                elev, azim_sun = calculate_solar_position(dt_series, lat, lon)
                poa = ghi_to_poa(ghi, elev, azim_sun, tilt, azim)
                dc_power = poa * dc_capacity_kw * efficiency / 1000.0
                ac_power = np.minimum(dc_power, ac_capacity_kw)
                return pd.Series(ac_power / ac_capacity_kw)

        cf_stats = calculate_annual_cf_stats(multi_year_data, compute_cf_for_year)
        multi_year_section = build_multi_year_stats_section(shape_stats, cf_stats, energy_type)

        # Update summary to reflect multi-year mode
        years_str = f"{min(fetched_years)}-{max(fetched_years)}" if fetched_years else ""
        summary = summary.replace(
            f"Year: {year_val}",
            f"Viewing: {year_val} of {len(fetched_years)} years ({years_str})"
        )

        # Combine single-year and multi-year cards
        cards = html.Div([
            html.H6(
                f"Year {year_val} Statistics",
                style={"fontWeight": "600", "marginBottom": "12px", "color": "#5f6368"}
            ),
            single_year_cards,
            multi_year_section,
        ])
    else:
        cards = single_year_cards

    return fig_shape, cards, summary


@app.callback(
    Output("download-data", "data"),
    Output("status-text", "children", allow_duplicate=True),
    Input("download-btn", "n_clicks"),
    State("energy-data-store", "data"),
    # Wind inputs
    State("cut-in", "value"),
    State("rated-speed", "value"),
    State("cut-out", "value"),
    State("turbine-mw", "value"),
    State("max-plant-cf", "value"),
    State("hub-height", "value"),
    State("use-measured-shear", "value"),
    State("shear-exponent", "value"),
    State("apply-density-correction", "value"),
    # Solar inputs
    State("panel-tilt", "value"),
    State("panel-azimuth", "value"),
    State("ilr", "value"),
    State("system-efficiency", "value"),
    State("dc-capacity", "value"),
    # Common
    State("target-mwh", "value"),
    State("location-store", "data"),
    State("year-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, json_data, cin, rated_spd, cout, turb_mw, max_plant_cf,
                 hub_height, use_measured_shear, shear_exponent, apply_density_correction,
                 panel_tilt, panel_azimuth, ilr, sys_eff, dc_cap,
                 target_mwh, loc, year):
    """Separate callback for CSV download."""
    if not n_clicks:
        return dash.no_update, dash.no_update

    # If store is empty, try to pull the latest finished payload from fetch state
    if json_data is None:
        state = get_fetch_state()
        if state.get("status") == "running":
            return dash.no_update, "Download blocked: fetch still running, please wait."
        json_data = state.get("result") or state.get("last_result")
        if json_data is None:
            return dash.no_update, "No data to download yet. Fetch a profile first."

    try:
        payload = json_data
        df_json = payload
        lat_source = None
        lon_source = None
        year_source = None
        energy_type = "wind"
        if isinstance(payload, dict):
            df_json = payload.get("data")
            lat_source = payload.get("lat")
            lon_source = payload.get("lon")
            year_source = payload.get("year")
            energy_type = payload.get("energy_type", "wind")

        if not df_json:
            return dash.no_update, "Download failed: no data payload available."

        df = pd.read_json(StringIO(df_json), orient="split")
        df["datetime"] = pd.to_datetime(df["datetime"])

        lat = lat_source if lat_source is not None else loc.get("lat", default_lat)
        lon = lon_source if lon_source is not None else loc.get("lon", default_lon)
        year_val = year_source if year_source is not None else (year if year is not None else 2024)

        if energy_type == "solar":
            # Guard against None values for solar (dc_cap is computed automatically)
            if any(v is None for v in [panel_tilt, panel_azimuth, ilr, sys_eff, target_mwh]):
                return dash.no_update, "Download failed: missing solar settings."

            # Calculate solar power with an automatic DC capacity derived from the peak hour
            sys_eff_frac = sys_eff / 100.0
            target_mwh = (target_mwh or 0) * 1000  # convert GWh to MWh for scaling
            nominal_dc_kw = 1000.0
            ac_power_kw_nominal, poa = apply_solar_power(
                df["ghi_wm2"], df["datetime"], lat, lon,
                panel_tilt, panel_azimuth, ilr, nominal_dc_kw, sys_eff_frac
            )
            peak_ac_kw = float(np.max(ac_power_kw_nominal)) if len(ac_power_kw_nominal) else 0.0
            dc_cap_auto_kw = peak_ac_kw * ilr if peak_ac_kw > 0 else nominal_dc_kw

            ac_power_kw, poa = apply_solar_power(
                df["ghi_wm2"], df["datetime"], lat, lon,
                panel_tilt, panel_azimuth, ilr, dc_cap_auto_kw, sys_eff_frac
            )
            df["ac_power_kw"] = ac_power_kw
            df["poa_wm2"] = poa

            ac_capacity_kw = dc_cap_auto_kw / ilr
            df["cf"] = df["ac_power_kw"] / ac_capacity_kw

            avg_cf = df["cf"].mean()
            installed_kw_ac = target_mwh * 1000 / (8760 * avg_cf) if avg_cf > 0 else 0
            installed_mw_ac = installed_kw_ac / 1000
            df["scaled_mw"] = df["cf"] * installed_mw_ac
            df["scaled_mwh"] = df["scaled_mw"]

            # Build solar download dataframe
            df_download = df[["datetime", "ghi_wm2", "poa_wm2", "ac_power_kw", "cf", "scaled_mw", "scaled_mwh"]].copy()
            df_download["datetime_utc"] = (
                pd.to_datetime(df_download["datetime"], utc=True)
                .dt.tz_convert("UTC")
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            df_download.insert(0, "latitude", lat)
            df_download.insert(1, "longitude", lon)
            filename = f"solar_hourly_shape_{year_val}_lat{lat:.2f}_lon{lon:.2f}.csv"
        else:
            # Wind processing
            if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf, hub_height, target_mwh]):
                return dash.no_update, "Download failed: missing turbine or purchase inputs."

            hub_h = hub_height or 100

            # Determine shear exponent: use measured from ERA5 or manual input
            if use_measured_shear and "shear_exponent" in df.columns:
                shear_values = df["shear_exponent"].values
            else:
                shear_values = shear_exponent or 0.14

            # Extrapolate wind speed from 100m to hub height using power law
            df["wind_speed_hub"] = extrapolate_wind_speed(df["wind_speed_100m"].values, hub_h, shear_values)

            # Convert max_plant_cf from percentage to fraction
            max_cf_frac = (max_plant_cf or 90) / 100.0
            df["power_mw_raw"] = apply_power_curve(
                df["wind_speed_hub"].values, cin, rated_spd, cout, turb_mw, max_cf_frac
            )

            # Apply air density correction if enabled
            if apply_density_correction and "air_density" in df.columns:
                df["power_mw"] = adjust_power_for_density(df["power_mw_raw"].values, df["air_density"].values)
            else:
                df["power_mw"] = df["power_mw_raw"]

            df["cf"] = df["power_mw"] / turb_mw

            avg_cf = df["cf"].mean()
            target_mwh = (target_mwh or 0) * 1000  # convert GWh to MWh for scaling
            installed_mw = target_mwh / (8760 * avg_cf) if avg_cf > 0 else 0
            df["scaled_mw"] = df["cf"] * installed_mw
            df["scaled_mwh"] = df["scaled_mw"]

            # Build wind download dataframe with shear and density data
            cols = ["datetime", "wind_speed_100m", "wind_speed_hub", "power_mw", "cf", "scaled_mw"]
            if "shear_exponent" in df.columns:
                cols.insert(3, "shear_exponent")
            if "air_density" in df.columns:
                cols.insert(4 if "shear_exponent" in df.columns else 3, "air_density")
            df_download = df[cols].copy()
            df_download["datetime_utc"] = (
                pd.to_datetime(df_download["datetime"], utc=True)
                .dt.tz_convert("UTC")
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            df_download.insert(0, "latitude", lat)
            df_download.insert(1, "longitude", lon)
            filename = f"wind_hourly_shape_{year_val}_lat{lat:.2f}_lon{lon:.2f}.csv"

        download = dcc.send_data_frame(
            df_download.to_csv,
            filename,
            index=False,
        )
        return download, "CSV ready — if your browser blocked it, allow the download."
    except Exception as exc:
        log_gee_error("download_csv", exc)
        return dash.no_update, f"Download error: {exc}"


# ---- Curtailment Analysis Callbacks ----


@app.callback(
    Output("curtailment-section", "style"),
    Output("curtailment-curve-graph", "figure"),
    Input("energy-data-store", "data"),
    # Wind inputs needed to compute CF
    Input("cut-in", "value"),
    Input("rated-speed", "value"),
    Input("cut-out", "value"),
    Input("turbine-mw", "value"),
    Input("max-plant-cf", "value"),
    Input("hub-height", "value"),
    Input("use-measured-shear", "value"),
    Input("shear-exponent", "value"),
    Input("apply-density-correction", "value"),
    # Solar inputs
    Input("panel-tilt", "value"),
    Input("panel-azimuth", "value"),
    Input("ilr", "value"),
    Input("system-efficiency", "value"),
    State("location-store", "data"),
    prevent_initial_call=True,
)
def update_curtailment_curve(json_data, cin, rated_spd, cout, turb_mw, max_plant_cf,
                              hub_height, use_measured_shear, shear_exponent, apply_density_correction,
                              panel_tilt, panel_azimuth, ilr, sys_eff, loc):
    """Update the curtailment curve graph when data is loaded."""
    if not json_data:
        return {"display": "none"}, go.Figure()

    try:
        # Handle dict format from energy-data-store
        if isinstance(json_data, dict):
            df_json = json_data.get("data")
            energy_type = json_data.get("energy_type", "wind")
            lat = json_data.get("lat", loc.get("lat", default_lat))
            lon = json_data.get("lon", loc.get("lon", default_lon))
        else:
            df_json = json_data
            energy_type = "wind"
            lat = loc.get("lat", default_lat)
            lon = loc.get("lon", default_lon)

        if not df_json:
            return {"display": "none"}, go.Figure()

        df = pd.read_json(StringIO(df_json), orient="split")
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Compute CF based on energy type
        if energy_type == "solar":
            if any(v is None for v in [panel_tilt, panel_azimuth, ilr, sys_eff]):
                return {"display": "none"}, go.Figure()
            if "ghi_wm2" not in df.columns:
                return {"display": "none"}, go.Figure()
            sys_eff_frac = sys_eff / 100.0
            nominal_dc_kw = 1000.0
            ac_power_kw, poa = apply_solar_power(
                df["ghi_wm2"], df["datetime"], lat, lon,
                panel_tilt, panel_azimuth, ilr, nominal_dc_kw, sys_eff_frac
            )
            ac_capacity_kw = nominal_dc_kw / ilr
            cf_series = ac_power_kw / ac_capacity_kw
        else:
            # Wind CF computation
            if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf]):
                return {"display": "none"}, go.Figure()
            if "wind_speed_100m" not in df.columns:
                return {"display": "none"}, go.Figure()

            hub_h = hub_height or 100
            if use_measured_shear and "shear_exponent" in df.columns:
                shear_values = df["shear_exponent"].values
            else:
                shear_values = shear_exponent or 0.14

            ws_hub = extrapolate_wind_speed(df["wind_speed_100m"].values, hub_h, shear_values)
            max_cf_frac = (max_plant_cf or 90) / 100.0
            power_mw = apply_power_curve(ws_hub, cin, rated_spd, cout, turb_mw, max_cf_frac)

            if apply_density_correction and "air_density" in df.columns:
                power_mw = adjust_power_for_density(power_mw, df["air_density"].values)

            cf_series = power_mw / turb_mw
        curve_data = calculate_curtailment_curve(cf_series)

        if curve_data is None:
            return {"display": "none"}, go.Figure()

        # Google-style font settings
        google_font = "Roboto, Open Sans, Helvetica Neue, sans-serif"
        title_font = dict(family=google_font, size=14, color="#202124", weight=600)
        axis_font = dict(family=google_font, size=12, color="#5f6368")
        tick_font = dict(family=google_font, size=11, color="#5f6368")

        # Set color based on energy type
        if energy_type == "solar":
            primary_color = google_colors["yellow"]
            secondary_color = google_colors["green"]
        else:
            primary_color = google_colors["blue"]
            secondary_color = google_colors["red"]

        # Create the curtailment curve figure
        # X-axis is Cap Level % (100% = peak, 0% = nothing) - goes from 100 down to 0
        fig = go.Figure()

        # Matching % line (primary metric)
        fig.add_trace(
            go.Scatter(
                x=curve_data["cap_pct"],  # Use cap level, not curtailment
                y=curve_data["matching_pct"],
                mode="lines",
                name="Hourly Matching %",
                line=dict(color=primary_color, width=2.5),
                hovertemplate="Cap: %{x:.0f}%<br>Matching: %{y:.1f}%<extra></extra>",
            )
        )

        # Energy delivered % line (secondary metric)
        fig.add_trace(
            go.Scatter(
                x=curve_data["cap_pct"],  # Use cap level, not curtailment
                y=curve_data["energy_delivered_pct"],
                mode="lines",
                name="Energy Retained %",
                line=dict(color=secondary_color, width=2, dash="dash"),
                hovertemplate="Cap: %{x:.0f}%<br>Energy: %{y:.1f}%<extra></extra>",
            )
        )

        # Add baseline CF annotation at 100% cap
        baseline_cf_pct = curve_data["baseline_cf"] * 100
        fig.add_annotation(
            x=100,  # At 100% cap level
            y=baseline_cf_pct,
            text=f"Capacity Factor: {baseline_cf_pct:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#5f6368",
            ax=-60,
            ay=0,
            font=dict(size=11, color="#5f6368"),
        )

        fig.update_layout(
            title=dict(text="Cap Level vs Hourly Matching", font=title_font),
            xaxis=dict(
                title="Cap Level (% of peak)",
                title_font=axis_font,
                tickfont=tick_font,
                gridcolor="#e8eaed",
                autorange="reversed",  # 100% on left, 0% on right
                range=[0, 100],
            ),
            yaxis=dict(
                title="Percentage",
                title_font=axis_font,
                tickfont=tick_font,
                gridcolor="#e8eaed",
                range=[0, 105],
            ),
            template="plotly_white",
            font=dict(family=google_font),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=50, b=40, l=60, r=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )

        return {"display": "block"}, fig

    except Exception as exc:
        log_gee_error("update_curtailment_curve", exc)
        return {"display": "none"}, go.Figure()


@app.callback(
    Output("curtailment-stats", "children"),
    Input("curtailment-slider", "value"),
    Input("energy-data-store", "data"),
    # Wind inputs needed to compute CF
    Input("cut-in", "value"),
    Input("rated-speed", "value"),
    Input("cut-out", "value"),
    Input("turbine-mw", "value"),
    Input("max-plant-cf", "value"),
    Input("hub-height", "value"),
    Input("use-measured-shear", "value"),
    Input("shear-exponent", "value"),
    Input("apply-density-correction", "value"),
    # Solar inputs
    Input("panel-tilt", "value"),
    Input("panel-azimuth", "value"),
    Input("ilr", "value"),
    Input("system-efficiency", "value"),
    State("location-store", "data"),
    prevent_initial_call=True,
)
def update_curtailment_stats(curtailment_pct, json_data, cin, rated_spd, cout, turb_mw, max_plant_cf,
                              hub_height, use_measured_shear, shear_exponent, apply_density_correction,
                              panel_tilt, panel_azimuth, ilr, sys_eff, loc):
    """Update stats display when curtailment slider is moved."""
    if not json_data or curtailment_pct is None:
        return html.Div()

    try:
        # Handle dict format from energy-data-store
        if isinstance(json_data, dict):
            df_json = json_data.get("data")
            energy_type = json_data.get("energy_type", "wind")
            lat = json_data.get("lat", loc.get("lat", default_lat))
            lon = json_data.get("lon", loc.get("lon", default_lon))
        else:
            df_json = json_data
            energy_type = "wind"
            lat = loc.get("lat", default_lat)
            lon = loc.get("lon", default_lon)

        if not df_json:
            return html.Div()

        df = pd.read_json(StringIO(df_json), orient="split")
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Compute CF based on energy type
        if energy_type == "solar":
            if any(v is None for v in [panel_tilt, panel_azimuth, ilr, sys_eff]):
                return html.Div()
            if "ghi_wm2" not in df.columns:
                return html.Div()
            sys_eff_frac = sys_eff / 100.0
            nominal_dc_kw = 1000.0
            ac_power_kw, poa = apply_solar_power(
                df["ghi_wm2"], df["datetime"], lat, lon,
                panel_tilt, panel_azimuth, ilr, nominal_dc_kw, sys_eff_frac
            )
            ac_capacity_kw = nominal_dc_kw / ilr
            cf_series = ac_power_kw / ac_capacity_kw
        else:
            # Wind CF computation
            if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf]):
                return html.Div()
            if "wind_speed_100m" not in df.columns:
                return html.Div()

            hub_h = hub_height or 100
            if use_measured_shear and "shear_exponent" in df.columns:
                shear_values = df["shear_exponent"].values
            else:
                shear_values = shear_exponent or 0.14

            ws_hub = extrapolate_wind_speed(df["wind_speed_100m"].values, hub_h, shear_values)
            max_cf_frac = (max_plant_cf or 90) / 100.0
            power_mw = apply_power_curve(ws_hub, cin, rated_spd, cout, turb_mw, max_cf_frac)

            if apply_density_correction and "air_density" in df.columns:
                power_mw = adjust_power_for_density(power_mw, df["air_density"].values)

            cf_series = power_mw / turb_mw

        cf_series = np.asarray(cf_series)
        cf_series = cf_series[np.isfinite(cf_series)]

        if cf_series.size == 0:
            return html.Div()

        # Calculate metrics at the selected cap level
        # Slider value is now directly the cap level (100% = peak, 0% = nothing)
        cap_pct = curtailment_pct if curtailment_pct is not None else 100
        cap_fraction = cap_pct / 100.0

        # Curtailed CF
        curtailed_cf = np.minimum(cf_series, cap_fraction)
        baseline_cf = np.mean(cf_series)
        curtailed_avg_cf = np.mean(curtailed_cf)

        # Energy retained
        energy_retained = (curtailed_avg_cf / baseline_cf * 100) if baseline_cf > 0 else 0

        # Matching %
        if cap_fraction > 0:
            matching = np.sum(curtailed_cf) / (cap_fraction * len(cf_series)) * 100
        else:
            matching = 100.0

        # Build stats display
        stats = [
            html.Div(
                [
                    html.Div(
                        "Production Cap",
                        style={"fontSize": "12px", "color": "#5f6368", "textTransform": "uppercase"},
                    ),
                    html.Div(
                        f"{cap_pct:.0f}%",
                        style={"fontSize": "24px", "fontWeight": "700", "color": "#202124"},
                    ),
                    html.Div(
                        "of nameplate capacity",
                        style={"fontSize": "11px", "color": "#5f6368"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        "Hourly Matching",
                        style={"fontSize": "12px", "color": "#5f6368", "textTransform": "uppercase"},
                    ),
                    html.Div(
                        f"{matching:.1f}%",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "color": google_colors["blue"] if energy_type != "solar" else google_colors["yellow"],
                        },
                    ),
                    html.Div(
                        "of baseload-equivalent box",
                        style={"fontSize": "11px", "color": "#5f6368"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        "Energy Retained",
                        style={"fontSize": "12px", "color": "#5f6368", "textTransform": "uppercase"},
                    ),
                    html.Div(
                        f"{energy_retained:.1f}%",
                        style={
                            "fontSize": "24px",
                            "fontWeight": "700",
                            "color": google_colors["green"] if energy_retained > 80 else google_colors["red"],
                        },
                    ),
                    html.Div(
                        "of uncurtailed generation",
                        style={"fontSize": "11px", "color": "#5f6368"},
                    ),
                ],
            ),
        ]

        return html.Div(stats)

    except Exception as exc:
        log_gee_error("update_curtailment_stats", exc)
        return html.Div()


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
