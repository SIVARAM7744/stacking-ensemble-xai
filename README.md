# Ensemble-Based Hybrid Disaster Prediction System

Full-stack project with:
- `ml_pipeline/` for model training + artifact generation
- `backend/app/` FastAPI backend (primary API)
- `src/` React + Vite frontend

## Current Status
- Core API + frontend integration exists.
- Disaster-specific model loading is enabled (`Flood` and `Earthquake`).
- Live provider ingestion is enabled (`/predict/live`) with offline fallback support.
- Local history persistence works out of the box using SQLite (default).

## Project Structure
- `backend/app/main.py`: API entrypoint
- `backend/app/predict_router.py`: prediction + DRI + persistence flow
- `backend/app/history_router.py`: prediction history endpoints
- `backend/app/explainability_router.py`: feature importance endpoint
- `ml_pipeline/train_pipeline.py`: training orchestrator
- `src/pages/*`: dashboard UI pages

## Environment Setup

### Quick Windows Setup (Recommended)
Use the provided command scripts so you do not need PowerShell activation:

```bat
.\setup.cmd
.\start_all.cmd
```

Unified Python dependency file is available at project root:

```bat
python -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

Or run separately:

```bat
.\start_backend.cmd
.\start_frontend.cmd
```

This avoids `Activate.ps1` execution policy issues and uses `npm.cmd` explicitly.
See full portable instructions: `PORTABLE_SETUP.md`.

### Traditional Two-Terminal Run
Open two terminals in the project root.

Terminal 1:
```bat
.\start_backend.cmd
```

Terminal 2:
```bat
.\start_frontend.cmd
```

Open:
- Frontend: `http://127.0.0.1:5173`
- Backend docs: `http://127.0.0.1:8000/docs`

Notes:
- Do not run `Activate.ps1`; the scripts call `.\.venv\Scripts\python.exe` directly.
- In PowerShell, local scripts must be prefixed with `.\`.
- Because this project path contains `#`, `start_frontend.cmd` mirrors the frontend to a safe runtime path before launching Vite.

### 1. Frontend
```bash
npm install
npm run dev
```

Frontend env:
- copy `.env.example` to `.env`
- set `VITE_API_BASE_URL`

### 2. Backend
```bash
cd backend
pip install -r requirements.txt
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend env:
- copy `backend/.env.example` to `backend/.env` (or export env vars directly)

### 3. Database Mode
Default mode is SQLite (no setup required).  
To switch to MySQL, set `DB_BACKEND=mysql` in `backend/.env` and create DB first:
```sql
CREATE DATABASE disaster_prediction_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

## Training (Pending by Plan)
Training is the only pending track right now.

When you are ready:
```bash
cd ml_pipeline
pip install -r requirements.txt
python train_pipeline.py
```

Expected output artifacts in `ml_pipeline/models/`:
- `preprocessing.pkl`
- `rf_model.pkl`
- `gb_model.pkl`
- `svm_model.pkl`
- `stacking_model.pkl`
- `metrics.json`
- `feature_importance.json`

## API Endpoints
- `GET /health`
- `GET /models/status`
- `POST /predict`
- `POST /predict/from-raw`
- `POST /predict/live`
- `GET /history`
- `GET /explainability`

## Live Data Providers (Integrated)
- Flood/weather signals: Open-Meteo
- Earthquake seismic signals: USGS

`POST /predict/live` accepts lat/lon and fetches provider data automatically.
If providers are unavailable, fallback feature values are used unless `strict_live_sources=true`.

## Notes
- `backend/app` is the active backend path.
- If frontend path contains `#`, Vite may behave inconsistently. A clean path without `#` is recommended.

## Cloud Deployment
Recommended split:

- frontend on Vercel
- backend on Railway

Deployment steps and required variables are documented in:

- [`VERCEL_DEPLOY.md`](./VERCEL_DEPLOY.md)
- [`RAILWAY_DEPLOY.md`](./RAILWAY_DEPLOY.md)
