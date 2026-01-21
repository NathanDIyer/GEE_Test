/**
 * Hourly matching calculations for combining resources against flat load.
 */

/**
 * Calculate flat hourly load from annual GWh.
 * @param {number} annualGwh - Annual energy in GWh
 * @returns {number} Hourly load in MWh
 */
export function calculateFlatLoad(annualGwh) {
  return (annualGwh * 1000) / 8760
}

/**
 * Scale hourly capacity factors to match target annual GWh.
 * @param {number[]} hourlyCf - Array of hourly capacity factors (0-1)
 * @param {number} targetGwh - Target annual energy in GWh
 * @returns {number[]} Array of hourly MWh values
 */
export function scaleToGwh(hourlyCf, targetGwh) {
  const totalCf = hourlyCf.reduce((a, b) => a + b, 0)
  const avgCf = totalCf / hourlyCf.length

  // Target MWh per hour = totalGwh * 1000 / 8760 / avgCf
  // Then multiply by CF for each hour
  if (avgCf === 0) return hourlyCf.map(() => 0)

  const impliedCapacity = (targetGwh * 1000) / (avgCf * 8760)

  return hourlyCf.map(cf => cf * impliedCapacity)
}

/**
 * Combine multiple resource profiles into a single combined profile.
 * @param {Object[]} resources - Array of resource objects with hourlyMwh arrays
 * @returns {number[]} Combined hourly MWh array (8760 values)
 */
export function combineProfiles(resources) {
  if (!resources.length) return new Array(8760).fill(0)

  const combined = new Array(8760).fill(0)

  for (const resource of resources) {
    if (!resource.hourlyMwh) continue
    for (let i = 0; i < 8760; i++) {
      combined[i] += resource.hourlyMwh[i] || 0
    }
  }

  return combined
}

/**
 * Calculate hourly matching metrics.
 * @param {number[]} generationMwh - Hourly generation MWh array
 * @param {number} hourlyLoadMwh - Flat hourly load in MWh
 * @returns {Object} Matching metrics
 */
export function calculateMatchMetrics(generationMwh, hourlyLoadMwh) {
  let totalMatched = 0
  let totalExcess = 0
  let totalGeneration = 0
  const annualLoad = hourlyLoadMwh * 8760

  const hourlyMatch = []
  const hourlyExcess = []

  for (let i = 0; i < 8760; i++) {
    const gen = generationMwh[i] || 0
    const matched = Math.min(gen, hourlyLoadMwh)
    const excess = Math.max(gen - hourlyLoadMwh, 0)

    hourlyMatch.push(matched)
    hourlyExcess.push(excess)

    totalMatched += matched
    totalExcess += excess
    totalGeneration += gen
  }

  const matchPct = annualLoad > 0 ? (totalMatched / annualLoad) * 100 : 0
  const excessPct = totalGeneration > 0 ? (totalExcess / totalGeneration) * 100 : 0

  return {
    totalMatchedGwh: totalMatched / 1000,
    totalExcessGwh: totalExcess / 1000,
    totalGenerationGwh: totalGeneration / 1000,
    annualLoadGwh: annualLoad / 1000,
    matchPct,
    excessPct,
    hourlyMatch,
    hourlyExcess,
  }
}

/**
 * Generate CSV content from analysis data.
 * @param {Object} data - Analysis data
 * @returns {string} CSV content
 */
export function generateCsv(data) {
  const { resources, hourlyLoadMwh, combinedMwh, hourlyMatch, hourlyExcess } = data

  // Build header
  const headers = ['hour', 'load_mwh']
  for (const resource of resources) {
    const key = `${resource.type}_${resource.lat}_${resource.lon}_mwh`
    headers.push(key)
  }
  headers.push('combined_mwh', 'matched_mwh', 'excess_mwh')

  // Build rows
  const rows = [headers.join(',')]

  for (let i = 0; i < 8760; i++) {
    const row = [i, hourlyLoadMwh.toFixed(2)]

    for (const resource of resources) {
      row.push((resource.hourlyMwh?.[i] || 0).toFixed(2))
    }

    row.push(
      (combinedMwh[i] || 0).toFixed(2),
      (hourlyMatch[i] || 0).toFixed(2),
      (hourlyExcess[i] || 0).toFixed(2)
    )

    rows.push(row.join(','))
  }

  return rows.join('\n')
}

/**
 * Download a file with the given content.
 * @param {string} content - File content
 * @param {string} filename - File name
 * @param {string} mimeType - MIME type
 */
export function downloadFile(content, filename, mimeType = 'text/csv') {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
