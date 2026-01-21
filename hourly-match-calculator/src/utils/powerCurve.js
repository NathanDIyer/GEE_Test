/**
 * Power curve calculations for wind and solar resources.
 * Ported from Python implementation in google_wind_dash.py
 */

// Standard air density at sea level, 15C (kg/m3)
const STANDARD_AIR_DENSITY = 1.225

/**
 * Extrapolate wind speed from 100m to hub height using power law.
 * Formula: ws_hub = ws_100m * (hub_height / 100)^shear_exponent
 */
export function extrapolateWindSpeed(ws100m, hubHeight, shearExponent = 0.14) {
  if (hubHeight === 100) return ws100m
  return ws100m * Math.pow(hubHeight / 100, shearExponent)
}

/**
 * Adjust power output for air density variations.
 * Wind turbine power is proportional to air density: P = 1/2 * rho * A * v^3
 */
export function adjustPowerForDensity(power, airDensity, applyCorrection = true) {
  if (!applyCorrection) return power
  return power * (airDensity / STANDARD_AIR_DENSITY)
}

/**
 * Apply wind power curve to calculate capacity factor.
 * Cubic ramp from cut-in to rated speed, then flat at rated power.
 */
export function applyWindPowerCurve(windSpeeds, cutIn, ratedSpeed, cutOut, maxCf = 0.90) {
  return windSpeeds.map(ws => {
    if (ws < cutIn || ws > cutOut) return 0
    if (ws >= ratedSpeed) return maxCf
    // Cubic ramp
    const fraction = Math.pow((ws - cutIn) / (ratedSpeed - cutIn), 3)
    return Math.min(fraction, maxCf)
  })
}

/**
 * Find the rated speed that produces a target capacity factor.
 * Used for CF-calibrated mode.
 */
export function findRatedSpeedForCF(windSpeeds, targetCF, cutIn = 3.0, maxCf = 0.90) {
  const searchMin = 7.0
  const searchMax = 18.0
  const searchStep = 0.1

  let bestRated = 12.0
  let bestError = Infinity

  for (let rated = searchMin; rated <= searchMax; rated += searchStep) {
    const cfs = applyWindPowerCurve(windSpeeds, cutIn, rated, 25.0, maxCf)
    const meanCF = cfs.reduce((a, b) => a + b, 0) / cfs.length
    const error = Math.abs(meanCF - targetCF)

    if (error < bestError) {
      bestError = error
      bestRated = rated
    }
  }

  return bestRated
}

/**
 * Calculate solar position (elevation, azimuth) for a datetime.
 */
export function calculateSolarPosition(datetimeUtc, lat, lon) {
  const date = new Date(datetimeUtc)
  const dayOfYear = getDayOfYear(date)
  const hour = date.getUTCHours() + date.getUTCMinutes() / 60.0

  // Solar declination (degrees)
  const declination = 23.45 * Math.sin(toRadians(360 / 365 * (284 + dayOfYear)))

  // Hour angle (degrees) - solar noon = 0
  const solarTime = hour + lon / 15.0
  const hourAngle = 15.0 * (solarTime - 12.0)

  // Solar elevation angle
  const latRad = toRadians(lat)
  const decRad = toRadians(declination)
  const haRad = toRadians(hourAngle)

  const sinElev = Math.sin(latRad) * Math.sin(decRad) +
                  Math.cos(latRad) * Math.cos(decRad) * Math.cos(haRad)
  const elevation = toDegrees(Math.asin(clamp(sinElev, -1, 1)))

  // Solar azimuth (degrees from north, clockwise)
  const elevRad = toRadians(elevation)
  const cosAz = (Math.sin(decRad) - Math.sin(latRad) * sinElev) /
                (Math.cos(latRad) * Math.cos(elevRad) + 1e-10)
  let azimuth = toDegrees(Math.acos(clamp(cosAz, -1, 1)))

  // Correct azimuth for afternoon (hour angle > 0)
  if (hourAngle > 0) azimuth = 360 - azimuth

  return { elevation, azimuth }
}

/**
 * Convert GHI to POA (Plane of Array) irradiance.
 * Simple model assuming mostly direct normal irradiance.
 */
export function ghiToPoa(ghi, solarElevation, solarAzimuth, panelTilt, panelAzimuth) {
  if (solarElevation <= 0) return 0

  const elevRad = toRadians(solarElevation)
  const solAzRad = toRadians(solarAzimuth)
  const tiltRad = toRadians(panelTilt)
  const panAzRad = toRadians(panelAzimuth)

  // Angle of incidence on tilted surface
  let cosAoi = Math.sin(elevRad) * Math.cos(tiltRad) +
               Math.cos(elevRad) * Math.sin(tiltRad) * Math.cos(solAzRad - panAzRad)
  cosAoi = clamp(cosAoi, 0, 1)

  // Diffuse fraction estimate (~20% at clear sky)
  const diffuseFrac = 0.2
  let sinElev = Math.sin(elevRad)
  sinElev = Math.max(sinElev, 0.05) // Avoid division issues at low sun

  // Direct component adjusted for tilt
  const directHorizontal = ghi * (1 - diffuseFrac)
  const directPoa = directHorizontal * cosAoi / sinElev

  // Diffuse component (isotropic sky model)
  const diffuse = ghi * diffuseFrac * (1 + Math.cos(tiltRad)) / 2

  // Ground reflected (albedo ~0.2)
  const groundReflected = ghi * 0.2 * (1 - Math.cos(tiltRad)) / 2

  const poa = directPoa + diffuse + groundReflected

  // Cap POA at reasonable maximum
  return clamp(poa, 0, 1300)
}

/**
 * Calculate solar capacity factor from GHI data.
 */
export function applySolarPower(ghiValues, datetimes, lat, lon, panelTilt, panelAzimuth, systemEfficiency = 0.86) {
  return ghiValues.map((ghi, i) => {
    const { elevation, azimuth } = calculateSolarPosition(datetimes[i], lat, lon)
    const poa = ghiToPoa(ghi, elevation, azimuth, panelTilt, panelAzimuth)

    // Capacity factor = POA * efficiency / reference irradiance (1000 W/m2)
    return (poa * systemEfficiency) / 1000.0
  })
}

// Helper functions
function toRadians(degrees) {
  return degrees * Math.PI / 180
}

function toDegrees(radians) {
  return radians * 180 / Math.PI
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value))
}

function getDayOfYear(date) {
  const start = new Date(date.getUTCFullYear(), 0, 0)
  const diff = date - start
  const oneDay = 1000 * 60 * 60 * 24
  return Math.floor(diff / oneDay)
}
