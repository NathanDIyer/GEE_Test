/**
 * Fetch solar data from Google Earth Engine.
 * Returns hourly capacity factors for a given location and year.
 */

import ee from '@google/earthengine'

// Simple in-memory cache (persists within serverless instance lifetime)
const cache = new Map()

function getCacheKey(lat, lon, year, cf) {
  const latKey = Math.round(lat * 100) / 100
  const lonKey = Math.round(lon * 100) / 100
  const cfKey = cf ? Math.round(cf * 100) / 100 : 'blind'
  return `solar_${latKey}_${lonKey}_${year}_${cfKey}`
}

// Constants
const CHUNK_WORKERS = 20

// Initialize Earth Engine with service account
let eeInitialized = false

async function initEE() {
  if (eeInitialized) return

  const privateKey = JSON.parse(process.env.GEE_PRIVATE_KEY || '{}')
  const serviceAccount = process.env.GEE_SERVICE_ACCOUNT

  if (!privateKey.private_key || !serviceAccount) {
    throw new Error('GEE credentials not configured')
  }

  return new Promise((resolve, reject) => {
    ee.data.authenticateViaPrivateKey(
      { ...privateKey, client_email: serviceAccount },
      () => {
        ee.initialize(null, null, () => {
          eeInitialized = true
          resolve()
        }, reject)
      },
      reject
    )
  })
}

// Fetch a slice of ERA5 solar data
async function fetchSolarSlice(lat, lon, startIso, hours) {
  const point = ee.Geometry.Point([lon, lat])
  const startDate = new Date(startIso)
  const endDate = new Date(startDate.getTime() + hours * 60 * 60 * 1000)

  const collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    .filterBounds(point)
    .filterDate(startDate.toISOString(), endDate.toISOString())
    .select(['surface_solar_radiation_downwards_hourly'])

  return new Promise((resolve, reject) => {
    collection.getRegion(point, 11132).evaluate((data, error) => {
      if (error) {
        reject(error)
        return
      }

      if (!data || data.length <= 1) {
        resolve([])
        return
      }

      const [header, ...rows] = data
      const ssrdIdx = header.indexOf('surface_solar_radiation_downwards_hourly')
      const timeIdx = header.indexOf('time')

      const result = rows.map(row => ({
        time: row[timeIdx],
        // Convert J/m2 (accumulated over hour) to W/m2 (average)
        ghi: row[ssrdIdx] / 3600.0,
      })).sort((a, b) => a.time - b.time)

      resolve(result)
    })
  })
}

// Fetch full year of solar data in chunks
async function fetchSolarYear(lat, lon, year) {
  const startDate = new Date(`${year}-01-01T00:00:00Z`)
  const endDate = new Date(`${year + 1}-01-01T00:00:00Z`)
  const totalHours = (endDate - startDate) / (1000 * 60 * 60)

  const numChunks = CHUNK_WORKERS
  const hoursPerChunk = totalHours / numChunks

  const chunks = []
  for (let i = 0; i < numChunks; i++) {
    const chunkStart = new Date(startDate.getTime() + i * hoursPerChunk * 60 * 60 * 1000)
    const chunkHours = i === numChunks - 1
      ? (endDate - chunkStart) / (1000 * 60 * 60)
      : hoursPerChunk
    chunks.push({ start: chunkStart.toISOString(), hours: chunkHours })
  }

  // Fetch all chunks in parallel
  const results = await Promise.all(
    chunks.map(chunk => fetchSolarSlice(lat, lon, chunk.start, chunk.hours))
  )

  // Flatten and sort
  const allData = results.flat().sort((a, b) => a.time - b.time)

  return allData
}

// Calculate solar position
function calculateSolarPosition(timestamp, lat, lon) {
  const date = new Date(timestamp)
  const dayOfYear = getDayOfYear(date)
  const hour = date.getUTCHours() + date.getUTCMinutes() / 60.0

  // Solar declination (degrees)
  const declination = 23.45 * Math.sin(toRadians(360 / 365 * (284 + dayOfYear)))

  // Hour angle
  const solarTime = hour + lon / 15.0
  const hourAngle = 15.0 * (solarTime - 12.0)

  // Solar elevation
  const latRad = toRadians(lat)
  const decRad = toRadians(declination)
  const haRad = toRadians(hourAngle)

  const sinElev = Math.sin(latRad) * Math.sin(decRad) +
                  Math.cos(latRad) * Math.cos(decRad) * Math.cos(haRad)
  const elevation = toDegrees(Math.asin(clamp(sinElev, -1, 1)))

  // Solar azimuth
  const elevRad = toRadians(elevation)
  const cosAz = (Math.sin(decRad) - Math.sin(latRad) * sinElev) /
                (Math.cos(latRad) * Math.cos(elevRad) + 1e-10)
  let azimuth = toDegrees(Math.acos(clamp(cosAz, -1, 1)))
  if (hourAngle > 0) azimuth = 360 - azimuth

  return { elevation, azimuth }
}

// Convert GHI to POA
function ghiToPoa(ghi, solarElevation, solarAzimuth, panelTilt, panelAzimuth) {
  if (solarElevation <= 0 || ghi <= 0) return 0

  const elevRad = toRadians(solarElevation)
  const solAzRad = toRadians(solarAzimuth)
  const tiltRad = toRadians(panelTilt)
  const panAzRad = toRadians(panelAzimuth)

  let cosAoi = Math.sin(elevRad) * Math.cos(tiltRad) +
               Math.cos(elevRad) * Math.sin(tiltRad) * Math.cos(solAzRad - panAzRad)
  cosAoi = clamp(cosAoi, 0, 1)

  const diffuseFrac = 0.2
  let sinElev = Math.sin(elevRad)
  sinElev = Math.max(sinElev, 0.05)

  const directHorizontal = ghi * (1 - diffuseFrac)
  const directPoa = directHorizontal * cosAoi / sinElev

  const diffuse = ghi * diffuseFrac * (1 + Math.cos(tiltRad)) / 2
  const groundReflected = ghi * 0.2 * (1 - Math.cos(tiltRad)) / 2

  const poa = directPoa + diffuse + groundReflected

  return clamp(poa, 0, 1300)
}

// Apply solar power model
function applySolarPower(solarData, lat, lon, panelTilt, panelAzimuth, systemEfficiency = 0.86) {
  return solarData.map(d => {
    const { elevation, azimuth } = calculateSolarPosition(d.time, lat, lon)
    const poa = ghiToPoa(d.ghi, elevation, azimuth, panelTilt, panelAzimuth)
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

// Calculate optimal tilt based on latitude (simple rule of thumb)
function getOptimalTilt(lat) {
  // Optimal tilt is roughly equal to latitude for annual energy
  return Math.abs(lat)
}

// Calculate optimal azimuth based on hemisphere
function getOptimalAzimuth(lat) {
  // South-facing in northern hemisphere, north-facing in southern
  return lat >= 0 ? 180 : 0
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { lat, lon, year, gwh, cf } = req.body

    // Validate inputs
    if (typeof lat !== 'number' || lat < -90 || lat > 90) {
      return res.status(400).json({ error: 'Invalid latitude' })
    }
    if (typeof lon !== 'number' || lon < -180 || lon > 180) {
      return res.status(400).json({ error: 'Invalid longitude' })
    }
    if (!year || year < 2020 || year > 2024) {
      return res.status(400).json({ error: 'Invalid year (must be 2020-2024)' })
    }

    // Check cache first
    const cacheKey = getCacheKey(lat, lon, year, cf)
    if (cache.has(cacheKey)) {
      console.log(`Cache hit: ${cacheKey}`)
      return res.status(200).json(cache.get(cacheKey))
    }

    console.log(`Cache miss: ${cacheKey}, fetching from GEE...`)

    // Initialize Earth Engine
    await initEE()

    // Fetch solar data
    const solarData = await fetchSolarYear(lat, lon, year)

    if (solarData.length < 8000) {
      return res.status(500).json({ error: 'Insufficient data returned from GEE' })
    }

    // Determine panel orientation
    const panelTilt = getOptimalTilt(lat)
    const panelAzimuth = getOptimalAzimuth(lat)

    // Default system efficiency
    let systemEfficiency = 0.86

    // Apply solar power model
    let hourlyCf = applySolarPower(solarData, lat, lon, panelTilt, panelAzimuth, systemEfficiency)

    // If CF is provided, calibrate the efficiency
    let method = 'blind_default'
    if (cf && cf > 0 && cf <= 1) {
      const currentAvgCf = hourlyCf.reduce((a, b) => a + b, 0) / hourlyCf.length
      if (currentAvgCf > 0) {
        // Scale the output to match target CF
        const scaleFactor = cf / currentAvgCf
        hourlyCf = hourlyCf.map(c => c * scaleFactor)
        method = 'cf_calibrated'
      }
    }

    // Trim or pad to exactly 8760 hours
    if (hourlyCf.length > 8760) {
      hourlyCf = hourlyCf.slice(0, 8760)
    } else if (hourlyCf.length < 8760) {
      const padding = new Array(8760 - hourlyCf.length).fill(0)
      hourlyCf = [...hourlyCf, ...padding]
    }

    // Calculate average CF
    const avgCf = hourlyCf.reduce((a, b) => a + b, 0) / hourlyCf.length

    const result = {
      hourly_cf: hourlyCf,
      avg_cf: avgCf,
      method,
    }

    // Store in cache
    cache.set(cacheKey, result)
    console.log(`Cached: ${cacheKey}`)

    return res.status(200).json(result)
  } catch (error) {
    console.error('Solar fetch error:', error)
    return res.status(500).json({ error: error.message || 'Failed to fetch solar data' })
  }
}
