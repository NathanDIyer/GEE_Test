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
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
import ee

# Password protection
from dash_auth import BasicAuth

CHUNK_HOURS = None  # computed dynamically per year
CHUNK_WORKERS = 20  # number of concurrent chunk fetches (year hours / 20)

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


def add_wind_speed(image):
    u = image.select("u_component_of_wind_100m")
    v = image.select("v_component_of_wind_100m")
    ws = u.hypot(v).rename("wind_speed_100m")
    return image.addBands(ws).copyProperties(image, image.propertyNames())


def fetch_era5_slice(lat, lon, start_iso, hours):
    """Pull a small ERA5 slice and return a sorted DataFrame."""
    start = pd.to_datetime(start_iso, utc=True)
    end = start + pd.to_timedelta(hours, unit="h")
    point = ee.Geometry.Point([lon, lat])

    coll = (
        ee.ImageCollection("ECMWF/ERA5/HOURLY")
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
        .select(["u_component_of_wind_100m", "v_component_of_wind_100m"])
        .map(add_wind_speed)
        .select(
            ["u_component_of_wind_100m", "v_component_of_wind_100m", "wind_speed_100m"]
        )
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
            "u_component_of_wind_100m",
            "v_component_of_wind_100m",
            "wind_speed_100m",
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
    """Chunked fetch for a full year (parallelized)."""
    start_iso = f"{year}-01-01T00:00:00Z"
    end_iso = f"{year + 1}-01-01T00:00:00Z"
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)
    worker_count = max_workers or CHUNK_WORKERS
    worker_count = max(1, worker_count)

    total_hours = (end - start) / pd.Timedelta(hours=1)
    # Split the year evenly across workers; last chunk closes the year to avoid drift.
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
        df_chunk = fetch_era5_slice(lat, lon, chunk_start, hours)
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
                                                            striped=True,
                                                            animated=True,
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
                                                    "Outputs",
                                                    style={"fontWeight": "700", "color": google_colors["blue"]},
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
                                        dcc.Graph(id="monthly-graph", style={"height": "320px"}),
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
                dcc.Interval(id="progress-interval", interval=1000, n_intervals=0),
                dcc.Store(id="location-store", data={"lat": default_lat, "lon": default_lon}),
                dcc.Store(id="year-store", data=2024),
                dcc.Store(id="energy-type-store", data="wind"),
                dcc.Store(id="energy-data-store"),
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
    Input("fetch-btn", "n_clicks"),
    State("location-store", "data"),
    State("year-dropdown", "value"),
    State("energy-type-store", "data"),
    prevent_initial_call=True,
)
def start_fetch(n, loc, year, energy_type):
    if not n:
        return dash.no_update, dash.no_update, dash.no_update
    state = get_fetch_state()
    if state["status"] == "running":
        return "Fetch already in progress...", dash.no_update, dash.no_update
    energy_type = energy_type or "wind"
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
    type_label = "solar" if energy_type == "solar" else "wind"
    return (
        f"Started {type_label} fetch for {year} (about a minute). Progress will update below.",
        year,
        None,
    )


@app.callback(
    Output("progress-bar", "value"),
    Output("progress-bar", "label"),
    Output("progress-bar", "color"),
    Output("status-text", "children", allow_duplicate=True),
    Output("energy-data-store", "data"),
    Input("progress-interval", "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def poll_progress(_):
    state = get_fetch_state()
    status = state["status"]
    val = 0
    label = "Idle"
    color = "success"
    status_text = state.get("message", "")
    data_out = dash.no_update

    if status == "running":
        total = max(state.get("total", 1), 1)
        progress = min(state.get("progress", 0), total)
        val = int((progress / total) * 100)
        label = f"{progress}/{total}"
        color = "info"
    elif status == "done":
        val = 100
        label = "Done"
        color = "success"
        data_out = state.get("result")
        status_text = state.get("message", "Done")
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

    return val, label, color, status_text, data_out


@app.callback(
    Output("shape-graph", "figure"),
    Output("monthly-graph", "figure"),
    Output("summary-text", "children"),
    Input("energy-data-store", "data"),
    # Wind inputs
    Input("cut-in", "value"),
    Input("rated-speed", "value"),
    Input("cut-out", "value"),
    Input("turbine-mw", "value"),
    Input("max-plant-cf", "value"),
    # Solar inputs
    Input("panel-tilt", "value"),
    Input("panel-azimuth", "value"),
    Input("ilr", "value"),
    Input("system-efficiency", "value"),
    Input("dc-capacity", "value"),
    # Common
    Input("target-mwh", "value"),
    State("location-store", "data"),
    State("year-store", "data"),
    prevent_initial_call=True,
)
def update_analysis(json_data, cin, rated_spd, cout, turb_mw, max_plant_cf,
                    panel_tilt, panel_azimuth, ilr, sys_eff, dc_cap,
                    target_mwh, loc, year):
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

    df = pd.read_json(df_json, orient="split")
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
        if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf, target_mwh]):
            return dash.no_update, dash.no_update, dash.no_update

        # Convert max_plant_cf from percentage to fraction
        max_cf_frac = (max_plant_cf or 90) / 100.0
        df["power_mw"] = apply_power_curve(
            df["wind_speed_100m"].values, cin, rated_spd, cout, turb_mw, max_cf_frac
        )
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
            f"{installed_mw:.2f} MW. Rows: {len(df)}. "
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
    fig_shape.update_layout(
        title=dict(text=f"{title_prefix} hourly shape (scaled to purchased GWh)", font=title_font),
        yaxis_title="Power (MW)",
        template="plotly_white",
        font=dict(family=google_font),
        yaxis=dict(titlefont=axis_font, tickfont=tick_font, gridcolor="#e8eaed"),
        xaxis=dict(tickfont=tick_font, gridcolor="#e8eaed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=20),
    )

    # Monthly totals plot
    monthly = df.resample("M", on="datetime")["scaled_mwh"].sum().reset_index()
    monthly["month"] = monthly["datetime"].dt.strftime("%b")
    fig_monthly = px.bar(
        monthly,
        x="month",
        y="scaled_mwh",
        title=f"{title_prefix} monthly energy (MWh, scaled from GWh target)",
        color_discrete_sequence=[google_colors["green"]],
        labels={"scaled_mwh": "MWh", "month": "Month"},
    )
    fig_monthly.update_layout(
        title=dict(font=title_font),
        font=dict(family=google_font),
        yaxis=dict(titlefont=axis_font, tickfont=tick_font, gridcolor="#e8eaed"),
        xaxis=dict(titlefont=axis_font, tickfont=tick_font),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50, b=50, l=60, r=20),
    )

    return fig_shape, fig_monthly, summary


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

        df = pd.read_json(df_json, orient="split")
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
            if any(v is None for v in [cin, rated_spd, cout, turb_mw, max_plant_cf, target_mwh]):
                return dash.no_update, "Download failed: missing turbine or purchase inputs."

            # Convert max_plant_cf from percentage to fraction
            max_cf_frac = (max_plant_cf or 90) / 100.0
            df["power_mw"] = apply_power_curve(
                df["wind_speed_100m"].values, cin, rated_spd, cout, turb_mw, max_cf_frac
            )
            df["cf"] = df["power_mw"] / turb_mw

            avg_cf = df["cf"].mean()
            target_mwh = (target_mwh or 0) * 1000  # convert GWh to MWh for scaling
            installed_mw = target_mwh / (8760 * avg_cf) if avg_cf > 0 else 0
            df["scaled_mw"] = df["cf"] * installed_mw
            df["scaled_mwh"] = df["scaled_mw"]

            # Build wind download dataframe (scaled_mw and scaled_mwh are identical for hourly; keep scaled_mw only)
            df_download = df[["datetime", "wind_speed_100m", "power_mw", "cf", "scaled_mw"]].copy()
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


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
