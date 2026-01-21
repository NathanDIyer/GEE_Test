import { useState, useCallback, useMemo } from 'react'
import { useAuth } from './hooks/useAuth'
import LoginGate from './components/LoginGate'
import ResourceList, { createNewResource } from './components/ResourceList'
import MiniMap from './components/MiniMap'
import HourlyChart from './components/HourlyChart'
import SummaryMetrics from './components/SummaryMetrics'
import CSVDownload from './components/CSVDownload'
import {
  calculateFlatLoad,
  scaleToGwh,
  combineProfiles,
  calculateMatchMetrics,
} from './utils/calculations'

const YEARS = [2024, 2023, 2022, 2021, 2020]

// Client-side cache for GEE data (persists during session)
const dataCache = new Map()

function getCacheKey(type, lat, lon, year, cf) {
  // Round lat/lon to 2 decimals (~1km precision) for cache key
  const latKey = Math.round(lat * 100) / 100
  const lonKey = Math.round(lon * 100) / 100
  const cfKey = cf ? Math.round(cf * 100) / 100 : 'blind'
  return `${type}_${latKey}_${lonKey}_${year}_${cfKey}`
}

// Generate mock hourly CF data for testing without API
function generateMockProfile(type, lat, cf) {
  const hourlyCf = []
  const targetCf = cf || (type === 'wind' ? 0.35 : 0.22)

  for (let hour = 0; hour < 8760; hour++) {
    const dayOfYear = Math.floor(hour / 24)
    const hourOfDay = hour % 24

    if (type === 'solar') {
      // Solar: peaks midday, zero at night, seasonal variation
      const solarNoon = 12
      const dayLength = 10 + 4 * Math.sin((dayOfYear - 80) * 2 * Math.PI / 365)
      const sunrise = solarNoon - dayLength / 2
      const sunset = solarNoon + dayLength / 2

      if (hourOfDay >= sunrise && hourOfDay <= sunset) {
        const hourAngle = (hourOfDay - solarNoon) / (dayLength / 2)
        const base = Math.cos(hourAngle * Math.PI / 2) * 0.8
        const seasonal = 0.7 + 0.3 * Math.sin((dayOfYear - 80) * 2 * Math.PI / 365)
        hourlyCf.push(Math.max(0, base * seasonal * (0.9 + Math.random() * 0.2)))
      } else {
        hourlyCf.push(0)
      }
    } else {
      // Wind: variable with some diurnal and seasonal patterns
      const baseWind = 0.3 + 0.1 * Math.sin((dayOfYear - 30) * 2 * Math.PI / 365)
      const diurnal = 1 + 0.15 * Math.sin((hourOfDay - 14) * 2 * Math.PI / 24)
      const random = 0.5 + Math.random()
      hourlyCf.push(Math.min(0.95, Math.max(0, baseWind * diurnal * random)))
    }
  }

  // Scale to match target CF
  const actualMean = hourlyCf.reduce((a, b) => a + b, 0) / hourlyCf.length
  const scale = targetCf / actualMean
  return {
    hourly_cf: hourlyCf.map(cf => Math.min(0.95, cf * scale)),
    avg_cf: targetCf,
    method: 'mock_data'
  }
}

// Demo data for first-time users
const DEMO_RESOURCES = [
  {
    id: 1,
    type: 'wind',
    lat: 44.3,
    lon: -96.5,
    gwh: 300,
    cf: 0.46,
    status: null,
    hourlyMwh: null,
    actualCf: null,
  },
]

const DEMO_LOAD = 500

export default function App() {
  const { isAuthenticated, isLoading: authLoading, login, logout } = useAuth()

  // State
  const [year, setYear] = useState(2024)
  const [loadGwh, setLoadGwh] = useState(DEMO_LOAD)
  const [resources, setResources] = useState(DEMO_RESOURCES)
  const [isCalculating, setIsCalculating] = useState(false)
  const [error, setError] = useState(null)

  // Derived state
  const hourlyLoadMwh = useMemo(() => calculateFlatLoad(loadGwh || 0), [loadGwh])

  const combinedMwh = useMemo(() => {
    return combineProfiles(resources.filter(r => r.hourlyMwh))
  }, [resources])

  const matchMetrics = useMemo(() => {
    if (!combinedMwh.some(v => v > 0)) return null
    return calculateMatchMetrics(combinedMwh, hourlyLoadMwh)
  }, [combinedMwh, hourlyLoadMwh])

  const summaryMetrics = useMemo(() => {
    if (!matchMetrics) return {}
    const totalProcurementGwh = resources
      .filter(r => r.gwh)
      .reduce((sum, r) => sum + r.gwh, 0)
    return {
      totalProcurementGwh,
      annualLoadGwh: loadGwh,
      matchPct: matchMetrics.matchPct,
      excessPct: matchMetrics.excessPct,
      totalGenerationGwh: matchMetrics.totalGenerationGwh,
    }
  }, [matchMetrics, resources, loadGwh])

  const csvData = useMemo(() => {
    if (!matchMetrics) return null
    return {
      resources,
      hourlyLoadMwh,
      combinedMwh,
      hourlyMatch: matchMetrics.hourlyMatch,
      hourlyExcess: matchMetrics.hourlyExcess,
    }
  }, [resources, hourlyLoadMwh, combinedMwh, matchMetrics])

  // Handlers
  const handleClearAll = useCallback(() => {
    setResources([createNewResource(Date.now())])
    setLoadGwh('')
    setError(null)
  }, [])

  const fetchResourceData = useCallback(async (resource) => {
    // Check cache first
    const cacheKey = getCacheKey(resource.type, resource.lat, resource.lon, year, resource.cf)
    if (dataCache.has(cacheKey)) {
      console.log(`Cache hit: ${cacheKey}`)
      return dataCache.get(cacheKey)
    }

    console.log(`Cache miss: ${cacheKey}, fetching from GEE...`)
    const endpoint = resource.type === 'solar' ? '/api/fetch-solar' : '/api/fetch-wind'

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat: resource.lat,
        lon: resource.lon,
        year,
        gwh: resource.gwh,
        cf: resource.cf,
      }),
    })

    // Check if API exists (404 = API not available, use mock data)
    if (response.status === 404) {
      console.log('API not available, using mock data')
      const mockData = generateMockProfile(resource.type, resource.lat, resource.cf)
      dataCache.set(cacheKey, mockData)
      return mockData
    }

    if (!response.ok) {
      let errorMsg = 'Failed to fetch data'
      try {
        const data = await response.json()
        errorMsg = data.error || errorMsg
      } catch (e) {
        // Response wasn't JSON
      }
      throw new Error(errorMsg)
    }

    const data = await response.json()

    // Store in cache
    dataCache.set(cacheKey, data)
    console.log(`Cached: ${cacheKey}`)

    return data
  }, [year])

  const handleCalculate = useCallback(async () => {
    // Validate resources
    const validResources = resources.filter(r =>
      typeof r.lat === 'number' && !isNaN(r.lat) &&
      typeof r.lon === 'number' && !isNaN(r.lon) &&
      typeof r.gwh === 'number' && r.gwh > 0
    )

    if (validResources.length === 0) {
      setError('Please add at least one resource with valid location and GWh')
      return
    }

    if (!loadGwh || loadGwh <= 0) {
      setError('Please enter a valid annual load')
      return
    }

    setError(null)
    setIsCalculating(true)

    // Mark all resources as loading
    setResources(prev => prev.map(r => ({
      ...r,
      status: validResources.some(vr => vr.id === r.id) ? 'loading' : r.status,
    })))

    try {
      // Fetch data for each resource
      const results = await Promise.allSettled(
        validResources.map(async (resource) => {
          const data = await fetchResourceData(resource)
          return { resource, data }
        })
      )

      // Update resources with results
      setResources(prev => prev.map(r => {
        const result = results.find(res =>
          res.status === 'fulfilled' && res.value.resource.id === r.id
        )
        const errorResult = results.find(res =>
          res.status === 'rejected' && validResources.find(vr => vr.id === r.id)
        )

        if (result && result.status === 'fulfilled') {
          const { data } = result.value
          const hourlyMwh = scaleToGwh(data.hourly_cf, r.gwh)
          return {
            ...r,
            status: 'loaded',
            hourlyMwh,
            actualCf: data.avg_cf,
          }
        } else if (errorResult) {
          return {
            ...r,
            status: 'error',
            error: errorResult.reason?.message || 'Failed to load',
          }
        }
        return r
      }))
    } catch (err) {
      setError(err.message || 'Failed to calculate')
    } finally {
      setIsCalculating(false)
    }
  }, [resources, loadGwh, fetchResourceData])

  // Auth loading state
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  // Login gate
  if (!isAuthenticated) {
    return <LoginGate onLogin={login} />
  }

  // Main app
  return (
    <div className="h-screen flex flex-col overflow-hidden bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0">
        <div className="flex justify-between items-center">
          <h1 className="text-xl font-bold text-gray-800">Hourly Match Calculator</h1>
          <button
            onClick={logout}
            className="text-sm text-gray-500 hover:text-gray-700 transition"
          >
            Logout
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Left panel */}
        <div className="w-[380px] flex-shrink-0 border-r border-gray-200 bg-white p-4 flex flex-col overflow-hidden">
          {/* Year and Load inputs */}
          <div className="mb-4 pb-4 border-b border-gray-200">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Year</label>
                <select
                  value={year}
                  onChange={(e) => setYear(parseInt(e.target.value))}
                  disabled={isCalculating}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                >
                  {YEARS.map(y => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Load (GWh)</label>
                <input
                  type="number"
                  value={loadGwh}
                  onChange={(e) => setLoadGwh(e.target.value === '' ? '' : parseFloat(e.target.value))}
                  disabled={isCalculating}
                  min="0"
                  step="10"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                  placeholder="500"
                />
              </div>
            </div>
          </div>

          {/* Resource list */}
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <ResourceList
              resources={resources}
              onChange={setResources}
              disabled={isCalculating}
            />
          </div>

          {/* Mini map */}
          <div className="mt-4 mb-4">
            <MiniMap resources={resources} />
          </div>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleCalculate}
              disabled={isCalculating}
              className="flex-1 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
            >
              {isCalculating ? (
                <>
                  <svg className="w-5 h-5 spinner" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Calculating...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                  Calculate
                </>
              )}
            </button>
            <button
              onClick={handleClearAll}
              disabled={isCalculating}
              className="px-4 py-3 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition"
              title="Clear All"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>

          {/* Error display */}
          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="flex-1 p-6 flex flex-col min-h-0 overflow-hidden">
          {/* Chart */}
          <div className="flex-1 bg-white rounded-lg shadow-sm p-4 mb-4 min-h-0">
            <HourlyChart
              combinedMwh={combinedMwh}
              hourlyMatch={matchMetrics?.hourlyMatch}
              hourlyExcess={matchMetrics?.hourlyExcess}
              hourlyLoadMwh={hourlyLoadMwh}
            />
          </div>

          {/* Summary and download */}
          <div className="flex-shrink-0 flex gap-4 items-end">
            <div className="flex-1">
              <SummaryMetrics metrics={summaryMetrics} />
            </div>
            <div className="flex-shrink-0">
              <CSVDownload data={csvData} disabled={isCalculating} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
