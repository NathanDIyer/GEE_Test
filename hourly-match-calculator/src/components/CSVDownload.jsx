import { generateCsv, downloadFile } from '../utils/calculations'

export default function CSVDownload({ data, disabled }) {
  const handleDownload = () => {
    if (!data || !data.combinedMwh || data.combinedMwh.length === 0) return

    const csv = generateCsv(data)
    const date = new Date().toISOString().split('T')[0]
    const filename = `hourly_match_${date}.csv`

    downloadFile(csv, filename, 'text/csv')
  }

  const hasData = data && data.combinedMwh && data.combinedMwh.some(v => v > 0)

  return (
    <button
      onClick={handleDownload}
      disabled={disabled || !hasData}
      className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition"
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Download CSV
    </button>
  )
}
