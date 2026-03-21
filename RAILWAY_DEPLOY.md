# Railway Backend Deploy Guide

This repo is ready to deploy the backend to Railway from the repo root.

Recommended split:

- frontend on Vercel
- backend on Railway

The backend service must use the repo root as its source because it needs both:

- [`backend`](./backend)
- [`ml_pipeline`](./ml_pipeline)

## Before You Deploy

1. Push this full project to one GitHub repo.
2. Make sure the model artifacts are committed if you want immediate inference on Railway.

Required artifact folders already used by the backend:

- [`ml_pipeline/models_tuned_v4_flood`](./ml_pipeline/models_tuned_v4_flood)
- [`ml_pipeline/models_tuned_v3_earthquake`](./ml_pipeline/models_tuned_v3_earthquake)

## Railway Setup

### 1. Create a Railway Project

- In Railway, create a new project from your GitHub repo.
### 2. Create Backend Service

- Add another service from the same repo.
- Keep source root at repo root `/`.
- Add service variable:
  - `RAILWAY_DOCKERFILE_PATH=backend.Dockerfile`

Optional but recommended backend variables:

- `DB_BACKEND=sqlite`
- `SQLITE_PATH=/data/disaster_prediction.db`

After deployment, generate a public domain for this service.

Useful backend health endpoint:

- `/health`

Useful docs endpoint:

- `/docs`

## SQLite Persistence on Railway

If you want local-history persistence with SQLite:

1. Attach a Railway Volume to the backend service.
2. Mount it at:
   - `/data`
3. Keep:
   - `SQLITE_PATH=/data/disaster_prediction.db`

Without a volume, SQLite will still work, but data can be ephemeral.

## Railway Variables Summary

### Backend

- `RAILWAY_DOCKERFILE_PATH=backend.Dockerfile`
- `DB_BACKEND=sqlite`
- `SQLITE_PATH=/data/disaster_prediction.db`

Optional MySQL mode instead of SQLite:

- `DB_BACKEND=mysql`
- `DB_HOST=...`
- `DB_PORT=3306`
- `DB_USER=...`
- `DB_PASSWORD=...`
- `DB_NAME=...`

## Expected Public URLs

- Backend docs:
  - `https://<backend-domain>/docs`
- Backend health:
  - `https://<backend-domain>/health`

## Notes

- Backend runs on `0.0.0.0:$PORT` inside Railway.
- The backend Dockerfile is [`backend.Dockerfile`](./backend.Dockerfile).
- Use the backend's generated public domain as `VITE_API_BASE_URL` in Vercel.
