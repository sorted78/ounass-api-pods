# OUNASS Kubernetes Pod Forecasting API

ML-powered Kubernetes capacity planning for OUNASS e-commerce platform. Predicts optimal pod requirements based on business metrics (GMV, users, marketing spend).

## 🎯 Features

- **Smart Forecasting**: ML models predict frontend and backend pod requirements
- **Business-Driven**: Uses GMV, user traffic, and marketing spend as inputs  
- **Real-time API**: FastAPI-based REST endpoints for easy integration
- **Google Sheets Integration**: Pull historical data and budget forecasts from Google Sheets
- **High Accuracy**: Gradient Boosting models with R² > 0.90
- **Production Ready**: Docker support, logging, error handling

## 📊 How It Works

```
Business Metrics (Input)     ML Model          Pod Predictions (Output)
━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━━━━
• GMV                    →   Gradient      →   • Frontend Pods: 15
• Users                      Boosting          • Backend Pods: 10
• Marketing Cost             Regressor         • Total Pods: 25
• Date features                                • Confidence: 92%
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9-3.12
- Google Cloud account (for Sheets API)
- Google Sheet with historical data

### 1. Clone & Setup

```bash
git clone https://github.com/sorted78/ounass-api-pods.git
cd ounass-api-pods

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Google Sheets

<details>
<summary><b>📋 Click here for detailed Google Sheets setup</b></summary>

#### A. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project: "OUNASS Pod Forecasting"
3. Enable APIs:
   - Google Sheets API
   - Google Drive API

#### B. Create Service Account

1. Go to: APIs & Services → Credentials
2. Create Credentials → Service Account
3. Name: `ounass-pod-forecasting`
4. Create and download JSON key
5. Rename to `credentials.json`
6. Place in project root

#### C. Prepare Your Google Sheet

1. Create/open your Google Sheet
2. Required columns (exact names):
   ```
   Date | GMV | Users | Marketing_Cost | Frontend_Pods | Backend_Pods
   ```

3. Fill historical data (rows with pod counts):
   ```
   2024-01-01 | 1200000 | 18000 | 45000 | 10 | 6
   2024-01-02 | 1150000 | 17500 | 42000 | 9  | 6
   ...
   ```

4. Leave future rows empty (for predictions):
   ```
   2024-07-01 | 1900000 | 28500 | 67500 |  |
   2024-07-02 | 1950000 | 29200 | 69000 |  |
   ```

#### D. Share Sheet

1. Open `credentials.json`
2. Copy the `client_email` value
3. Share your Google Sheet with this email (Viewer permission)

#### E. Get Sheet ID

From your sheet URL:
```
https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit
                                        ^^^^^^^^^^^^
```

</details>

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your values
nano .env
```

Update these values:
```env
GOOGLE_SHEET_ID=your_actual_sheet_id_here
GOOGLE_CREDENTIALS_PATH=./credentials.json
```

### 4. Run the API

```bash
# Make run script executable
chmod +x run.sh

# Start API
./run.sh
```

API will be available at:
- **Main**: http://127.0.0.1:8000
- **Docs**: http://127.0.0.1:8000/docs  
- **Health**: http://127.0.0.1:8000/api/v1/health

### 5. Train & Predict

```bash
# Train the model
curl -X POST http://127.0.0.1:8000/api/v1/train

# Get tomorrow's forecast
curl http://127.0.0.1:8000/api/v1/forecast/daily

# Get next 7 days
curl "http://127.0.0.1:8000/api/v1/forecast/range?days=7"
```

## 📖 API Endpoints

### Train Model
```http
POST /api/v1/train
```
Trains ML models on historical data from Google Sheets.

**Response:**
```json
{
  "frontend_mae": 0.82,
  "frontend_rmse": 1.05,
  "frontend_r2": 0.94,
  "backend_mae": 0.65,
  "backend_rmse": 0.88,
  "backend_r2": 0.96
}
```

### Daily Forecast
```http
GET /api/v1/forecast/daily?target_date=2024-07-01
```
Get pod prediction for a specific date.

**Response:**
```json
{
  "date": "2024-07-01",
  "frontend_pods": 18,
  "backend_pods": 12,
  "total_pods": 30,
  "confidence_score": 0.92,
  "metrics": {
    "gmv": 1900000,
    "users": 28500,
    "marketing_cost": 67500
  }
}
```

### Range Forecast
```http
GET /api/v1/forecast/range?start_date=2024-07-01&days=7
```
Get predictions for multiple days.

### Health Check
```http
GET /api/v1/health
```

## 🐋 Docker Deployment

```bash
# Build image
docker build -t ounass-pod-forecasting .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/credentials.json:/app/credentials.json:ro \
  -e GOOGLE_SHEET_ID=your_sheet_id \
  --name ounass-api \
  ounass-pod-forecasting
```

## 🏗️ Project Structure

```
ounass-api-pods/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   └── endpoints.py        # API routes
│   ├── models/
│   │   └── forecasting.py      # ML models
│   └── services/
│       └── sheets_service.py   # Google Sheets integration
├── data/
│   └── sample_data.csv         # Sample dataset for testing
├── tests/
│   └── test_api.py             # API tests
├── .env.example                # Environment template
├── .gitignore
├── Dockerfile
├── requirements.txt
├── run.sh                      # Quick start script
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_SHEET_ID` | Google Sheet identifier | - | ✅ |
| `GOOGLE_CREDENTIALS_PATH` | Path to service account JSON | `./credentials.json` | ✅ |
| `API_HOST` | API host address | `0.0.0.0` | ❌ |
| `API_PORT` | API port number | `8000` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |

### Google Sheet Format

Your sheet must have these exact column names:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `Date` | Date | Format: YYYY-MM-DD | ✅ |
| `GMV` | Number | Gross Merchandise Value | ✅ |
| `Users` | Integer | Active users | ✅ |
| `Marketing_Cost` | Number | Marketing spend | ✅ |
| `Frontend_Pods` | Integer | Actual frontend pods (historical only) | For training |
| `Backend_Pods` | Integer | Actual backend pods (historical only) | For training |

## 🤖 ML Model Details

### Features Used
- **Base**: GMV, Users, Marketing_Cost, DayOfWeek, DayOfMonth, Month
- **Engineered**: GMV per User, Marketing per User, ROAS, IsWeekend

### Algorithm
- **Model**: Gradient Boosting Regressor
- **Separate models** for frontend and backend pods
- **Feature scaling**: StandardScaler for normalization

### Training Requirements
- Minimum 10 historical records (30+ recommended)
- Historical data must include actual pod counts

## 🚨 Troubleshooting

### "Spreadsheet not found" (404)
- Check `GOOGLE_SHEET_ID` in `.env`
- Ensure sheet is shared with service account email
- Verify both Google Sheets API and Google Drive API are enabled

### "Model not trained"
- Train model first: `curl -X POST http://127.0.0.1:8000/api/v1/train`
- Ensure you have at least 10 rows with pod data

### "No budget data found for date"
- Check date format in sheet (YYYY-MM-DD)
- Ensure requested date exists in your sheet

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please fork and submit a pull request.

---

**Made with ❤️ for efficient cloud resource management**
