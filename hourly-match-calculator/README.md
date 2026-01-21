# Hourly Match Calculator

A React application for analyzing hourly matching between renewable energy resources and flat load profiles. Uses Google Earth Engine ERA5 reanalysis data to generate realistic hourly capacity factor profiles.

## Features

- **Multi-resource support**: Add up to 5 wind or solar resources
- **GEE-powered profiles**: Hourly generation profiles from ERA5 reanalysis data
- **CF calibration**: Optionally provide target capacity factors for calibration
- **Interactive visualization**: 8760-hour chart showing matched vs excess generation
- **CSV export**: Download detailed hourly data for further analysis

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env.local` with your credentials:
   ```
   ACCESS_PASSWORD=your_password
   GEE_SERVICE_ACCOUNT=your-sa@project.iam.gserviceaccount.com
   GEE_PRIVATE_KEY={"type":"service_account","private_key":"..."}
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. For serverless functions locally, use Vercel CLI:
   ```bash
   npm i -g vercel
   vercel dev
   ```

### Deployment to Vercel

1. Connect your repository to Vercel
2. Add environment variables in the Vercel dashboard:
   - `ACCESS_PASSWORD` - Password for the login gate
   - `GEE_SERVICE_ACCOUNT` - Google Earth Engine service account email
   - `GEE_PRIVATE_KEY` - Full JSON key for the service account
3. Deploy

## Architecture

```
├── src/
│   ├── components/
│   │   ├── LoginGate.jsx      # Password authentication
│   │   ├── ResourceCard.jsx   # Individual resource inputs
│   │   ├── ResourceList.jsx   # Resource management
│   │   ├── MiniMap.jsx        # Location visualization
│   │   ├── HourlyChart.jsx    # 8760 chart with Recharts
│   │   ├── SummaryMetrics.jsx # Key metrics display
│   │   └── CSVDownload.jsx    # Export functionality
│   ├── hooks/
│   │   └── useAuth.js         # Authentication state
│   └── utils/
│       ├── powerCurve.js      # Wind/solar power calculations
│       └── calculations.js    # Matching calculations
└── api/
    ├── auth.js                # Password validation
    ├── fetch-wind.js          # Wind GEE data
    ├── fetch-solar.js         # Solar GEE data
    └── health.js              # Health check
```

## GEE Service Account Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Create a service account in your project
3. Grant it access to Earth Engine
4. Create and download a JSON key
5. Register the service account with Earth Engine at [signup.earthengine.google.com](https://signup.earthengine.google.com/#!/service_accounts)

## Input Specifications

### Resources (max 5)
- **Type**: Wind or Solar
- **Latitude**: -90 to 90
- **Longitude**: -180 to 180
- **Annual GWh**: Target procurement amount
- **CF (optional)**: Target capacity factor for calibration

### Load
- **Annual GWh**: Flat load assumption (hourly = annual / 8760)

### Year
- 2020-2024 (ERA5 data availability)

## Output Metrics

| Metric | Formula |
|--------|---------|
| Hourly Match % | Sum(min(gen, load)) / Total Load × 100 |
| Excess % | Sum(max(gen - load, 0)) / Total Gen × 100 |

## Technology Stack

- **Frontend**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Maps**: react-leaflet
- **Backend**: Vercel Serverless Functions
- **Data**: Google Earth Engine (ERA5/ERA5-Land)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ACCESS_PASSWORD` | Simple password gate |
| `GEE_SERVICE_ACCOUNT` | GEE service account email |
| `GEE_PRIVATE_KEY` | GEE service account JSON key |

## License

MIT
