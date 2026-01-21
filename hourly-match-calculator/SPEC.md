# Hourly Match Calculator - Specification

## Overview

A single-page React application that allows users to quickly estimate hourly matching between renewable resources and a flat load profile. Users input their procurement details (location, type, GWh) and the tool generates hourly profiles using GEE ERA5 data with validated blind defaults.

**Key Architecture Decision:** React frontend + Vercel serverless functions (not Python/Dash) for easier hosting and deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VERCEL                                  │
├─────────────────────────────┬───────────────────────────────────┤
│   STATIC (React App)        │   SERVERLESS (/api)              │
│                             │                                   │
│   - Login gate              │   /api/fetch-wind                 │
│   - Resource input UI       │   /api/fetch-solar                │
│   - Chart visualization     │   /api/health                     │
│   - CSV export              │                                   │
│                             │   ┌─────────────────────────┐     │
│                             │   │ GEE Service Account     │     │
│                             │   │ (env variable)          │     │
│                             │   └─────────────────────────┘     │
└─────────────────────────────┴───────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18 |
| Build | Vite |
| Styling | Tailwind CSS (or CSS modules) |
| Charts | Recharts |
| Hosting | Vercel |
| Backend | Vercel Serverless Functions (Node.js) |
| GEE | earthengine-api (Node.js) |
| Auth | Simple password gate |

---

## Authentication

### Simple Password Gate

- Single shared password protects the entire app
- Password stored as environment variable (`ACCESS_PASSWORD`)
- No user accounts or registration
- Session persists via localStorage (optional: sessionStorage for stricter)

**Flow:**
1. User visits site → sees login screen
2. User enters password
3. If correct → store flag in localStorage, show app
4. If incorrect → show error, stay on login
5. Logout button clears localStorage

**Implementation:**
```javascript
// Simple check - password compared client-side against hash
// Or: API route validates and returns session token
const isAuthenticated = localStorage.getItem('authenticated') === 'true';
```

**Security Note:** This is a simple gate for non-sensitive tools. Password is not high-security. For sensitive data, use proper auth (Firebase, Auth0, etc.).

---

## User Flow

1. User enters password on login screen
2. User enters the year (single year)
3. User enters their annual load (GWh)
4. User adds up to 5 resources via + button, each with:
   - Type: Wind or Solar
   - Location: Latitude, Longitude
   - Procurement: Annual GWh
   - (Optional) Capacity Factor
5. User clicks "Calculate"
6. Frontend calls `/api/fetch-wind` or `/api/fetch-solar` for each resource
7. Frontend combines profiles and renders chart
8. User can download CSV of all profiles

---

## Page Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  HOURLY MATCH CALCULATOR                          [Logout]      │
├───────────────────────────┬─────────────────────────────────────┤
│  LEFT PANEL (40%)         │  RIGHT PANEL (60%)                  │
│                           │                                     │
│  ┌─────────────────────┐  │  ┌─────────────────────────────────┐│
│  │ Year: [2024 ▼]      │  │  │                                 ││
│  │ Load: [____] GWh    │  │  │                                 ││
│  └─────────────────────┘  │  │                                 ││
│                           │  │     8760 HOURLY CHART           ││
│  RESOURCES                │  │                                 ││
│  ┌─────────────────────┐  │  │   - Combined profile            ││
│  │ ☀️ Solar            │  │  │   - Load line (flat)            ││
│  │ Lat: 35.2 Lon:-120.5│  │  │   - Match shading               ││
│  │ GWh: 200  CF: [opt] │  │  │   - Excess shading              ││
│  │               [🗑️]  │  │  │                                 ││
│  └─────────────────────┘  │  │                                 ││
│  ┌─────────────────────┐  │  │                                 ││
│  │ 💨 Wind             │  │  └─────────────────────────────────┘│
│  │ Lat: 44.3 Lon:-96.5 │  │                                     │
│  │ GWh: 300  CF: 0.46  │  │  ┌─────────────────────────────────┐│
│  │               [🗑️]  │  │  │ SUMMARY METRICS                 ││
│  └─────────────────────┘  │  │ Total Procurement: 500 GWh      ││
│                           │  │ Annual Load: 400 GWh            ││
│  [+ Add Resource]         │  │ Hourly Match: 72.3%             ││
│                           │  │ Excess: 27.7%                   ││
│  ┌─────────────────────┐  │  └─────────────────────────────────┘│
│  │      MINI MAP       │  │                                     │
│  │  (display only)     │  │  [📥 Download CSV]                  │
│  │   • markers         │  │                                     │
│  └─────────────────────┘  │                                     │
│                           │                                     │
│  [🔄 Calculate]           │                                     │
│                           │                                     │
└───────────────────────────┴─────────────────────────────────────┘
```

**Constraints:**
- Single page, no vertical scrolling
- Viewport height: 100vh
- Responsive but optimized for desktop

---

## API Endpoints (Vercel Serverless)

### POST /api/fetch-wind

**Request:**
```json
{
  "lat": 44.3,
  "lon": -96.5,
  "year": 2024,
  "gwh": 300,
  "cf": 0.46  // optional
}
```

**Response:**
```json
{
  "hourly_cf": [0.12, 0.34, 0.56, ...],  // 8760 values
  "avg_cf": 0.461,
  "method": "cf_calibrated"  // or "blind_default"
}
```

**Logic:**
1. Check cache (Vercel KV or file-based)
2. If not cached, fetch from GEE using service account
3. Apply power curve (blind or CF-calibrated)
4. Return hourly CF profile

### POST /api/fetch-solar

**Request:**
```json
{
  "lat": 35.2,
  "lon": -120.5,
  "year": 2024,
  "gwh": 200,
  "cf": null  // optional
}
```

**Response:**
```json
{
  "hourly_cf": [0.0, 0.0, 0.1, 0.4, ...],  // 8760 values
  "avg_cf": 0.22,
  "method": "blind_default"
}
```

### GET /api/health

Returns `{ "status": "ok" }` for monitoring.

---

## Power Curve Defaults (Blind Estimation)

### Wind (no CF provided)
```javascript
const cut_in = 3.0;   // m/s
const rated = 10.5;   // m/s
const exp = 2.5;      // physically realistic
// Scale output to match GWh
```

### Wind (CF provided)
```javascript
const cut_in = 3.0;
const exp = 2.5;
const rated = findRatedForCF(targetCF, windSpeeds); // iterate 7-14 m/s
const max_cf = targetCF / meanCF(rated);
```

### Solar
- Convert GHI to POA (plane of array)
- Apply system losses (default ~14%)
- If CF provided, calibrate losses to match

---

## Input Specifications

### Year Selector
- Dropdown: 2020, 2021, 2022, 2023, 2024
- Default: 2024

### Load Input
- Numeric input (GWh)
- Required
- Flat load: `hourly_mwh = gwh * 1000 / 8760`

### Resource Cards (max 5)

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| Type | Dropdown | Yes | Wind / Solar |
| Latitude | Number | Yes | -90 to 90 |
| Longitude | Number | Yes | -180 to 180 |
| GWh | Number | Yes | > 0 |
| CF | Number | No | 0.01 to 1.0 |

### Mini Map
- Library: react-leaflet or mapbox-gl
- Display-only (no click interaction)
- Wind markers: blue 💨
- Solar markers: yellow ☀️
- Auto-fit bounds to markers

---

## Output Specifications

### 8760 Hourly Chart

**Library:** Recharts

**X-axis:** Hour of year (0-8759)
**Y-axis:** Power (MW)

**Traces:**
1. Combined generation (area)
2. Load line (flat, dashed)
3. Match shading (green, where gen ≤ load)
4. Excess shading (orange, where gen > load)

### Summary Metrics

| Metric | Calculation |
|--------|-------------|
| Total Procurement | Σ resource GWh |
| Annual Load | User input |
| Hourly Match % | Σ min(gen, load) / load × 100 |
| Excess % | Σ max(gen - load, 0) / Σ gen × 100 |

### CSV Download

**Filename:** `hourly_match_YYYY-MM-DD.csv`

**Columns:**
- `hour` (0-8759)
- `load_mwh`
- `{type}_{lat}_{lon}_mwh` (per resource)
- `combined_mwh`
- `matched_mwh`
- `excess_mwh`

---

## Environment Variables (Vercel)

| Variable | Description |
|----------|-------------|
| `ACCESS_PASSWORD` | Simple gate password |
| `GEE_SERVICE_ACCOUNT` | GEE service account email |
| `GEE_PRIVATE_KEY` | GEE service account private key (JSON) |

---

## File Structure

```
hourly-match-calculator/
├── SPEC.md
├── README.md
├── package.json
├── vite.config.js
├── vercel.json
├── .env.local              # Local dev secrets
├── public/
│   └── index.html
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── components/
│   │   ├── LoginGate.jsx
│   │   ├── ResourceCard.jsx
│   │   ├── ResourceList.jsx
│   │   ├── MiniMap.jsx
│   │   ├── HourlyChart.jsx
│   │   ├── SummaryMetrics.jsx
│   │   └── CSVDownload.jsx
│   ├── hooks/
│   │   └── useAuth.js
│   ├── utils/
│   │   ├── powerCurve.js
│   │   └── calculations.js
│   └── styles/
│       └── index.css
└── api/
    ├── fetch-wind.js       # Vercel serverless function
    ├── fetch-solar.js
    └── health.js
```

---

## Deployment

### Vercel Setup

1. Connect GitHub repo to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main

### Local Development

```bash
npm install
npm run dev          # Start Vite dev server
vercel dev           # Test serverless functions locally
```

---

## Demo Mode

On first visit (after login), show a demo/example to illustrate how the tool works:

**Default Demo State:**
- Year: 2024
- Load: 500 GWh
- 1 Resource pre-filled:
  - Type: Wind
  - Lat: 44.3, Lon: -96.5 (South Dakota - validated site)
  - GWh: 300
  - CF: 0.46 (optional, can leave blank to demo blind mode)

**Behavior:**
- Demo data appears on first load
- "Clear All" button to start fresh
- User can modify demo data or add more resources

This ensures users always see at least one working example.

---

## Open Questions

1. Should results be cached per-user or globally? (Recommend: globally by lat/lon/year)
2. Timeout handling for slow GEE fetches (Vercel has 10s default, can extend to 60s)
3. Should we show a loading skeleton while fetching? (Recommend: yes)

---

## Acceptance Criteria

- [ ] Password gate prevents unauthorized access
- [ ] Single page loads without scrolling
- [ ] Can add/remove up to 5 resources
- [ ] Wind and solar types work correctly
- [ ] CF is optional; blind defaults used when not provided
- [ ] Chart shows combined profile vs flat load
- [ ] Match % and excess % displayed correctly
- [ ] CSV download includes all columns
- [ ] Serverless functions call GEE successfully
- [ ] Deploys to Vercel without errors
- [ ] No console errors in production

---

*Spec Version: 1.0*
*Created: January 2026*
