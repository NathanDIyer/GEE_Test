export default function SummaryMetrics({ metrics }) {
  const {
    totalProcurementGwh = 0,
    annualLoadGwh = 0,
    matchPct = 0,
    excessPct = 0,
    totalGenerationGwh = 0,
  } = metrics || {}

  const formatGwh = (value) => {
    if (!value && value !== 0) return '—'
    return value.toFixed(1)
  }

  const formatPct = (value) => {
    if (!value && value !== 0) return '—'
    return value.toFixed(1) + '%'
  }

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h3 className="font-semibold text-gray-700 mb-3">Summary Metrics</h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-lg p-3 shadow-sm">
          <p className="text-xs text-gray-500 mb-1">Total Procurement</p>
          <p className="text-lg font-semibold text-gray-800">
            {formatGwh(totalProcurementGwh)} <span className="text-sm font-normal">GWh</span>
          </p>
        </div>

        <div className="bg-white rounded-lg p-3 shadow-sm">
          <p className="text-xs text-gray-500 mb-1">Annual Load</p>
          <p className="text-lg font-semibold text-gray-800">
            {formatGwh(annualLoadGwh)} <span className="text-sm font-normal">GWh</span>
          </p>
        </div>

        <div className="bg-white rounded-lg p-3 shadow-sm border-l-4 border-green-500">
          <p className="text-xs text-gray-500 mb-1">Hourly Match</p>
          <p className="text-lg font-semibold text-green-600">
            {formatPct(matchPct)}
          </p>
        </div>

        <div className="bg-white rounded-lg p-3 shadow-sm border-l-4 border-orange-500">
          <p className="text-xs text-gray-500 mb-1">Excess Generation</p>
          <p className="text-lg font-semibold text-orange-600">
            {formatPct(excessPct)}
          </p>
        </div>
      </div>

      {totalGenerationGwh > 0 && (
        <div className="mt-3 text-xs text-gray-500 text-center">
          Total Generation: {formatGwh(totalGenerationGwh)} GWh
        </div>
      )}
    </div>
  )
}
