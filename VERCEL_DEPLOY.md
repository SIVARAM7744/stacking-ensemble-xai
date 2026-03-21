# Vercel Frontend Deploy Guide

This repo is ready to deploy the frontend to Vercel from the repo root.

## What Vercel Should Deploy

- Framework: Vite
- Root directory: `/`
- Build command: `npm run build`
- Output directory: `dist`

Vercel can auto-detect the framework, but these are the expected values.

## Required Environment Variable

Set this on the Vercel project before deploying:

- `VITE_API_BASE_URL=https://<your-railway-backend-domain>`

Example:

- `VITE_API_BASE_URL=https://stacking-ensemble-xai-backend.up.railway.app`

## Why `vercel.json` Exists

The frontend uses React Router with browser history routes such as:

- `/`
- `/predict`
- `/explainability`
- `/history`
- `/api-status`

Direct refreshes on those paths must rewrite back to `index.html`, so
[`vercel.json`](./vercel.json) is included for SPA routing.

## Vercel Setup Steps

1. Import the GitHub repo into Vercel.
2. Keep the root directory at `/`.
3. Set `VITE_API_BASE_URL` to your Railway backend public URL.
4. Deploy.

After deploy, open the Vercel URL and verify:

- the dashboard loads
- `/predict` opens directly
- `/explainability` opens directly
- `/api-status` shows the Railway backend as reachable

## Notes

- The frontend is fully static; Vercel only needs the built `dist/` output.
- If you change the Railway backend domain later, update `VITE_API_BASE_URL` and redeploy.
