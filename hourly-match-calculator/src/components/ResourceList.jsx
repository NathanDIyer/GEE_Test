import ResourceCard from './ResourceCard'

const MAX_RESOURCES = 5

// Default locations for new resources (good wind/solar sites in US)
const DEFAULT_LOCATIONS = [
  { lat: 44.3, lon: -96.5 },   // South Dakota (wind)
  { lat: 35.0, lon: -117.5 },  // California (solar)
  { lat: 41.5, lon: -100.5 },  // Nebraska (wind)
  { lat: 32.5, lon: -111.0 },  // Arizona (solar)
  { lat: 46.0, lon: -103.0 },  // North Dakota (wind)
]

const createNewResource = (id, index = 0) => ({
  id,
  type: index % 2 === 0 ? 'wind' : 'solar',  // Alternate wind/solar
  lat: DEFAULT_LOCATIONS[index % DEFAULT_LOCATIONS.length].lat,
  lon: DEFAULT_LOCATIONS[index % DEFAULT_LOCATIONS.length].lon,
  gwh: 300,
  cf: null,
  status: null,
  hourlyMwh: null,
  actualCf: null,
})

export default function ResourceList({ resources, onChange, disabled }) {
  const handleAddResource = () => {
    if (resources.length >= MAX_RESOURCES) return
    const newId = Date.now()
    onChange([...resources, createNewResource(newId, resources.length)])
  }

  const handleUpdateResource = (index, updated) => {
    const newResources = [...resources]
    newResources[index] = updated
    onChange(newResources)
  }

  const handleDeleteResource = (index) => {
    const newResources = resources.filter((_, i) => i !== index)
    onChange(newResources)
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-semibold text-gray-700 text-sm">Resources</h3>
        <button
          onClick={handleAddResource}
          disabled={disabled || resources.length >= MAX_RESOURCES}
          className="px-2 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center gap-1"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add ({resources.length}/{MAX_RESOURCES})
        </button>
      </div>

      <div className="space-y-1">
        {resources.length === 0 ? (
          <div className="text-center py-4 text-gray-400 text-sm border-2 border-dashed border-gray-200 rounded-lg">
            No resources. Click "Add" above.
          </div>
        ) : (
          resources.map((resource, index) => (
            <ResourceCard
              key={resource.id}
              resource={resource}
              onChange={(updated) => handleUpdateResource(index, updated)}
              onDelete={() => handleDeleteResource(index)}
              disabled={disabled}
            />
          ))
        )}
      </div>
    </div>
  )
}

export { createNewResource }
