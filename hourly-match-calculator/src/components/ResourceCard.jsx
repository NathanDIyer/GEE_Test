const RESOURCE_TYPES = [
  { value: 'wind', label: 'Wind', icon: '💨', color: 'bg-blue-50 border-blue-400' },
  { value: 'solar', label: 'Solar', icon: '☀️', color: 'bg-amber-50 border-amber-400' },
]

export default function ResourceCard({ resource, onChange, onDelete, disabled }) {
  const typeConfig = RESOURCE_TYPES.find(t => t.value === resource.type) || RESOURCE_TYPES[0]

  const handleChange = (field, value) => {
    onChange({ ...resource, [field]: value })
  }

  const handleNumberChange = (field, value) => {
    const num = value === '' ? '' : parseFloat(value)
    handleChange(field, num)
  }

  return (
    <div className={`flex items-center gap-2 p-2 rounded-lg border-l-4 ${typeConfig.color} mb-2`}>
      {/* Type icon & selector */}
      <select
        value={resource.type}
        onChange={(e) => handleChange('type', e.target.value)}
        disabled={disabled}
        className="bg-transparent text-sm font-medium w-20 border-none focus:ring-0 cursor-pointer disabled:cursor-not-allowed"
        title="Resource type"
      >
        {RESOURCE_TYPES.map(t => (
          <option key={t.value} value={t.value}>{t.icon} {t.label}</option>
        ))}
      </select>

      {/* Lat */}
      <input
        type="number"
        value={resource.lat}
        onChange={(e) => handleNumberChange('lat', e.target.value)}
        disabled={disabled}
        step="0.1"
        className="w-16 px-1.5 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
        placeholder="Lat"
        title="Latitude"
      />

      {/* Lon */}
      <input
        type="number"
        value={resource.lon}
        onChange={(e) => handleNumberChange('lon', e.target.value)}
        disabled={disabled}
        step="0.1"
        className="w-20 px-1.5 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
        placeholder="Lon"
        title="Longitude"
      />

      {/* GWh */}
      <div className="flex items-center gap-1">
        <input
          type="number"
          value={resource.gwh}
          onChange={(e) => handleNumberChange('gwh', e.target.value)}
          disabled={disabled}
          step="10"
          className="w-16 px-1.5 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
          placeholder="GWh"
          title="Annual GWh"
        />
        <span className="text-xs text-gray-500">GWh</span>
      </div>

      {/* CF (optional) */}
      <div className="flex items-center gap-1">
        <input
          type="number"
          value={resource.cf === null ? '' : resource.cf}
          onChange={(e) => handleNumberChange('cf', e.target.value || null)}
          disabled={disabled}
          step="0.01"
          min="0.01"
          max="1"
          className="w-20 px-1.5 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
          placeholder="CF"
          title="Capacity Factor (optional)"
        />
        <span className="text-xs text-gray-500">CF</span>
      </div>

      {/* Status indicator */}
      {resource.status === 'loaded' && (
        <span className="text-green-500 text-sm" title={`CF: ${(resource.actualCf * 100).toFixed(1)}%`}>✓</span>
      )}
      {resource.status === 'loading' && (
        <span className="text-gray-400 text-sm spinner">⏳</span>
      )}
      {resource.status === 'error' && (
        <span className="text-red-500 text-sm" title={resource.error}>✗</span>
      )}

      {/* Delete button */}
      <button
        onClick={onDelete}
        disabled={disabled}
        className="ml-auto text-gray-400 hover:text-red-500 disabled:opacity-50 transition p-1"
        title="Remove resource"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}
