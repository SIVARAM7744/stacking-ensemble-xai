# Railway Deploy Guide

This repo is ready to deploy to Railway as:

- 1 GitHub repo
- 1 Railway project
- 2 Railway services

Do not split this into 2 repos.

## Recommended Architecture

- `frontend` service
  - Source: this repo
  - Root directory: `/`
  - Dockerfile: root [`Dockerfile`](./Dockerfile)
- `backend` service
  - Source: this repo
  - Root directory: `/`
  - Custom Dockerfile path: `backend.Dockerfile`

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

### 2. Create Frontend Service

- Add a service from the same repo.
- Keep source root at repo root `/`.
- Railway will use the root [`Dockerfile`](./Dockerfile).

Set this variable on the frontend service:

- `VITE_API_BASE_URL=https://<your-backend-public-domain>`

After deployment, generate a public domain for this service.

### 3. Create Backend Service

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

### Frontend

- `VITE_API_BASE_URL=https://<backend-domain>`

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

- Frontend:
  - `https://<frontend-domain>`
- Backend docs:
  - `https://<backend-domain>/docs`
- Backend health:
  - `https://<backend-domain>/health`

## Notes

- Backend runs on `0.0.0.0:$PORT` inside Railway.
- Frontend is built with Vite and served as a static app.
- `VITE_API_BASE_URL` must point to the backend public URL, not a local URL.
- Browser clients cannot use Railway private service hostnames; use the backend's generated public domain for the frontend.
