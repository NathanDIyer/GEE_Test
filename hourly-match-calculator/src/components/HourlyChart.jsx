import { useMemo } from 'react'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

// Sample every N hours for performance (8760 points is too many)
const SAMPLE_RATE = 6 // Show every 6th hour = 1460 points

export default function HourlyChart({ combinedMwh, hourlyMatch, hourlyExcess, hourlyLoadMwh }) {
  const chartData = useMemo(() => {
    if (!combinedMwh || combinedMwh.length === 0) {
      // Empty placeholder data
      return Array.from({ length: 365 }, (_, i) => ({
        hour: i * 24,
        load: 0,
        matched: 0,
        excess: 0,
        generation: 0,
      }))
    }

    // Sample the data for performance
    const sampled = []
    for (let i = 0; i < 8760; i += SAMPLE_RATE) {
      sampled.push({
        hour: i,
        load: hourlyLoadMwh || 0,
        matched: hourlyMatch?.[i] || 0,
        excess: hourlyExcess?.[i] || 0,
        generation: combinedMwh[i] || 0,
      })
    }
    return sampled
  }, [combinedMwh, hourlyMatch, hourlyExcess, hourlyLoadMwh])

  const maxY = useMemo(() => {
    if (!combinedMwh || combinedMwh.length === 0) return 100
    const maxGen = Math.max(...combinedMwh)
    const maxLoad = hourlyLoadMwh || 0
    return Math.ceil(Math.max(maxGen, maxLoad) * 1.1)
  }, [combinedMwh, hourlyLoadMwh])

  const formatXAxis = (hour) => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    const dayOfYear = Math.floor(hour / 24)
    const monthIndex = Math.floor(dayOfYear / 30.4)
    return months[Math.min(monthIndex, 11)]
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null

    const hour = label
    const dayOfYear = Math.floor(hour / 24) + 1
    const hourOfDay = hour % 24

    return (
      <div className="bg-white p-3 rounded shadow-lg border text-sm">
        <p className="font-semibold mb-1">Day {dayOfYear}, Hour {hourOfDay}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(1)} MWh
          </p>
        ))}
      </div>
    )
  }

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={chartData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="hour"
            tickFormatter={formatXAxis}
            tick={{ fontSize: 11 }}
            interval={Math.floor(chartData.length / 12)}
          />
          <YAxis
            domain={[0, maxY]}
            tick={{ fontSize: 11 }}
            width={50}
            tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px' }}
            iconType="rect"
            iconSize={10}
          />

          {/* Matched generation (green area under load) */}
          <Area
            type="monotone"
            dataKey="matched"
            name="Matched"
            fill="#22c55e"
            fillOpacity={0.6}
            stroke="none"
            stackId="1"
          />

          {/* Excess generation (orange area above load) */}
          <Area
            type="monotone"
            dataKey="excess"
            name="Excess"
            fill="#f97316"
            fillOpacity={0.6}
            stroke="none"
            stackId="1"
          />

          {/* Load line (flat, dashed) */}
          <Line
            type="monotone"
            dataKey="load"
            name="Load"
            stroke="#6b7280"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
