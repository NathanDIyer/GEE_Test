/**
 * Fetch wind data from Google Earth Engine.
 * Returns hourly capacity factors for a given location and year.
 */

import ee from '@google/earthengine'

// Simple in-memory cache (persists within serverless instance lifetime)
// For persistent caching across instances, use Vercel KV
const cache = new Map()

function getCacheKey(lat, lon, year, cf) {
  const latKey = Math.round(lat * 100) / 100
  const lonKey = Math.round(lon * 100) / 100
  const cfKey = cf ? Math.round(cf * 100) / 100 : 'blind'
  return `wind_${latKey}_${lonKey}_${year}_${cfKey}`
}

// Constants for chunking (matching Python implementation)
const CHUNK_WORKERS = 20
const CHUNK_MULTIPLIER = 3
const STANDARD_AIR_DENSITY = 1.225

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

// Fetch a slice of ERA5 wind data
async function fetchWindSlice(lat, lon, startIso, hours) {
  const point = ee.Geometry.Point([lon, lat])
  const startDate = new Date(startIso)
  const endDate = new Date(startDate.getTime() + hours * 60 * 60 * 1000)

  const bands = [
    'u_component_of_wind_100m',
    'v_component_of_wind_100m',
    'temperature_2m',
    'surface_pressure',
  ]

  const collection = ee.ImageCollection('ECMWF/ERA5/HOURLY')
    .filterBounds(point)
    .filterDate(startDate.toISOString(), endDate.toISOString())
    .select(bands)

  // Add computed wind speed and air density
  const withMetrics = collection.map((image) => {
    const u100 = image.select('u_component_of_wind_100m')
    const v100 = image.select('v_component_of_wind_100m')
    const ws100 = u100.hypot(v100).rename('wind_speed_100m')

    const temp = image.select('temperature_2m')
    const pressure = image.select('surface_pressure')
    const density = pressure.divide(temp.multiply(287.05)).rename('air_density')

    return image.addBands([ws100, density])
  }).select(['wind_speed_100m', 'air_density'])

  return new Promise((resolve, reject) => {
    withMetrics.getRegion(point, 27000).evaluate((data, error) => {
      if (error) {
        reject(error)
        return
      }

      if (!data || data.length <= 1) {
        resolve([])
        return
      }

      const [header, ...rows] = data
      const wsIdx = header.indexOf('wind_speed_100m')
      const densityIdx = header.indexOf('air_density')
      const timeIdx = header.indexOf('time')

      const result = rows.map(row => ({
        time: row[timeIdx],
        windSpeed: row[wsIdx],
        airDensity: row[densityIdx],
      })).sort((a, b) => a.time - b.time)

      resolve(result)
    })
  })
}

// Fetch full year of wind data in chunks
async function fetchWindYear(lat, lon, year) {
  const startDate = new Date(`${year}-01-01T00:00:00Z`)
  const endDate = new Date(`${year + 1}-01-01T00:00:00Z`)
  const totalHours = (endDate - startDate) / (1000 * 60 * 60)

  const numChunks = CHUNK_WORKERS * CHUNK_MULTIPLIER
  const hoursPerChunk = totalHours / numChunks

  const chunks = []
  for (let i = 0; i < numChunks; i++) {
    const chunkStart = new Date(startDate.getTime() + i * hoursPerChunk * 60 * 60 * 1000)
    const chunkHours = i === numChunks - 1
      ? (endDate - chunkStart) / (1000 * 60 * 60)
      : hoursPerChunk
    chunks.push({ start: chunkStart.toISOString(), hours: chunkHours })
  }

  // Fetch chunks in parallel (limited concurrency)
  const results = []
  const batchSize = CHUNK_WORKERS

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize)
    const batchResults = await Promise.all(
      batch.map(chunk => fetchWindSlice(lat, lon, chunk.start, chunk.hours))
    )
    results.push(...batchResults.flat())
  }

  // Sort by time
  results.sort((a, b) => a.time - b.time)

  return results
}

// Apply wind power curve to calculate capacity factors
function applyWindPowerCurve(windSpeeds, cutIn, ratedSpeed, cutOut, maxCf = 0.90) {
  return windSpeeds.map(ws => {
    if (ws < cutIn || ws > cutOut) return 0
    if (ws >= ratedSpeed) return maxCf
    const fraction = Math.pow((ws - cutIn) / (ratedSpeed - cutIn), 3)
    return Math.min(fraction, maxCf)
  })
}

// Find rated speed that produces target CF
function findRatedSpeedForCF(windSpeeds, targetCF, cutIn = 3.0, maxCf = 0.90) {
  let bestRated = 12.0
  let bestError = Infinity

  for (let rated = 7.0; rated <= 18.0; rated += 0.1) {
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

// Adjust power for air density
function adjustForDensity(cfs, densities) {
  return cfs.map((cf, i) => {
    const density = densities[i] || STANDARD_AIR_DENSITY
    return cf * (density / STANDARD_AIR_DENSITY)
  })
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

    // Fetch wind data
    const windData = await fetchWindYear(lat, lon, year)

    if (windData.length < 8000) {
      return res.status(500).json({ error: 'Insufficient data returned from GEE' })
    }

    // Extract wind speeds and densities
    const windSpeeds = windData.map(d => d.windSpeed)
    const densities = windData.map(d => d.airDensity)

    // Apply power curve
    let hourlyCf
    let method

    const cutIn = 3.0
    const cutOut = 25.0
    const maxCf = 0.90

    if (cf && cf > 0 && cf <= 1) {
      // CF-calibrated mode
      const ratedSpeed = findRatedSpeedForCF(windSpeeds, cf, cutIn, maxCf)
      hourlyCf = applyWindPowerCurve(windSpeeds, cutIn, ratedSpeed, cutOut, maxCf)
      method = 'cf_calibrated'
    } else {
      // Blind default mode
      const ratedSpeed = 10.5 // Blind default
      hourlyCf = applyWindPowerCurve(windSpeeds, cutIn, ratedSpeed, cutOut, maxCf)
      method = 'blind_default'
    }

    // Adjust for air density
    hourlyCf = adjustForDensity(hourlyCf, densities)

    // Trim or pad to exactly 8760 hours
    if (hourlyCf.length > 8760) {
      hourlyCf = hourlyCf.slice(0, 8760)
    } else if (hourlyCf.length < 8760) {
      // Pad with zeros if needed
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
    console.error('Wind fetch error:', error)
    return res.status(500).json({ error: error.message || 'Failed to fetch wind data' })
  }
}
