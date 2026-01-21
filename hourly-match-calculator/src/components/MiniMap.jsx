import { useEffect, useMemo } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'

// Fix for Leaflet marker icons in Vite/webpack
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Custom icons for wind and solar
const createIcon = (color, emoji) => {
  return L.divIcon({
    className: 'custom-marker',
    html: `<div style="
      background: ${color};
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    ">${emoji}</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  })
}

const windIcon = createIcon('#3b82f6', '💨')
const solarIcon = createIcon('#f59e0b', '☀️')

// Component to auto-fit bounds when markers change
function FitBounds({ markers }) {
  const map = useMap()

  useEffect(() => {
    if (markers.length === 0) {
      // Default view of US
      map.setView([39.8, -98.5], 3)
      return
    }

    if (markers.length === 1) {
      map.setView([markers[0].lat, markers[0].lon], 5)
      return
    }

    const bounds = L.latLngBounds(markers.map(m => [m.lat, m.lon]))
    map.fitBounds(bounds, { padding: [30, 30] })
  }, [markers, map])

  return null
}

export default function MiniMap({ resources }) {
  // Filter resources with valid coordinates
  const validResources = useMemo(() => {
    return resources.filter(r =>
      typeof r.lat === 'number' && !isNaN(r.lat) &&
      typeof r.lon === 'number' && !isNaN(r.lon) &&
      r.lat >= -90 && r.lat <= 90 &&
      r.lon >= -180 && r.lon <= 180
    )
  }, [resources])

  return (
    <div className="h-40 rounded-lg overflow-hidden border border-gray-200">
      <MapContainer
        center={[39.8, -98.5]}
        zoom={3}
        className="h-full w-full"
        zoomControl={false}
        scrollWheelZoom={false}
        dragging={false}
        doubleClickZoom={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />

        <FitBounds markers={validResources} />

        {validResources.map((resource, index) => (
          <Marker
            key={resource.id || index}
            position={[resource.lat, resource.lon]}
            icon={resource.type === 'solar' ? solarIcon : windIcon}
          >
            <Popup>
              <div className="text-sm">
                <strong>{resource.type === 'solar' ? '☀️ Solar' : '💨 Wind'}</strong><br />
                {resource.lat.toFixed(2)}, {resource.lon.toFixed(2)}<br />
                {resource.gwh ? `${resource.gwh} GWh` : ''}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  )
}
