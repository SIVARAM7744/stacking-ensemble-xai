/**
 * api.ts
 * ------
 * Central HTTP service layer for the Ensemble-Based Hybrid Disaster
 * Prediction System frontend.
 *
 * All requests are routed through the typed `request<T>()` helper which:
 *   - Enforces a configurable timeout (default 8 s)
 *   - Parses FastAPI validation errors from the response body
 *   - Throws a plain Error so callers can display a user-facing message
 *
 * Backend base URL can be overridden at build time via:
 *   VITE_API_BASE_URL=http://my-server:8000
 */

export const API_BASE_URL: string =
  (import.meta as unknown as { env: Record<string, string> }).env
    ?.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ── Request / Response Types ─────────────────────────────────────────────

export interface PredictionRequest {
  disaster_type: string;
  rainfall?: number | null;
  temperature?: number | null;
  humidity?: number | null;
  soil_moisture?: number | null;
  wind_speed?: number | null;
  atmospheric_pressure?: number | null;
  seismic_activity?: number | null;
  previous_disaster: number;
  location?: string | null;
  region_code?: string | null;
}

export interface ModelProbabilities {
  random_forest: number;
  gradient_boosting: number;
  svm: number;
  stacking: number;
}

export interface PredictionResponse {
  predicted_disaster: string;
  probability: number;
  dri: number;
  risk_level: string;
  confidence: number;
  model_probabilities: ModelProbabilities;
}

export interface RawDataPredictionRequest {
  disaster_type: string;
  latitude?: number | null;
  longitude?: number | null;
  location?: string | null;
  region_code?: string | null;
  previous_disaster?: number;
  weather_payload?: Record<string, unknown>;
  seismic_payload?: Record<string, unknown>;
}

export interface RawDataPredictionResponse {
  mapped_features: Record<string, unknown>;
  prediction: PredictionResponse;
}

export interface LivePredictionRequest {
  disaster_type: string;
  latitude: number;
  longitude: number;
  location?: string | null;
  region_code?: string | null;
  previous_disaster?: number;
  earthquake_radius_km?: number;
  earthquake_min_magnitude?: number;
  strict_live_sources?: boolean;
}

export interface LivePredictionResponse {
  source: Record<string, unknown>;
  mapped_features: Record<string, unknown>;
  prediction: PredictionResponse;
}

export interface HealthResponse {
  status: string;
  api_version: string;
  db_connected: boolean;
  db_engine: string;
}

export interface ModelsStatusResponse {
  status: string;
  models: string[];
  message: string;
}

export interface HistoryRecord {
  id: string | null;
  timestamp: string | null;
  disaster_type: string;
  location: string | null;
  region_code: string | null;
  input: Record<string, number | null>;
  predicted_disaster: string | null;
  probability: number | null;
  dri: number | null;
  risk_level: string | null;
  confidence: number | null;
  model_probabilities: Record<string, number>;
}

export interface HistoryResponse {
  total: number;
  page: number;
  page_size: number;
  records: HistoryRecord[];
}

export interface HistoryQueryParams {
  page?: number;
  page_size?: number;
  disaster_type?: string;
  risk_level?: string;
  date_from?: string;
  date_to?: string;
}

// ── Explainability types ──────────────────────────────────────────────────

/** A single feature's global importance and signed SHAP-style contribution. */
export interface FeatureImportanceItem {
  /** Feature name, e.g. 'rainfall' */
  feature: string;
  /** Relative importance as a percentage (0–100). All items sum to 100. */
  importance_pct: number;
  /**
   * Signed SHAP-style contribution.
   * Positive  → increases risk
   * Negative  → decreases risk
   */
  shap_contribution: number;
}

/** Response from GET /explainability */
export interface ExplainabilityResponse {
  /** True when feature_importance.json was found and parsed successfully. */
  available: boolean;
  /** Filesystem path the data was loaded from (for transparency). */
  source: string;
  /** Features sorted by importance descending. */
  feature_importance: FeatureImportanceItem[];
  /** Feature with the highest importance score. */
  top_feature: string | null;
  /** Total number of features in the list. */
  total_features: number;
  /** Note about the explainability method used (SHAP vs approximation). */
  note: string;
}

export type ExplainabilityDisasterType = 'Flood' | 'Earthquake';

// ── Core HTTP helper ──────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options?: RequestInit,
  timeoutMs = 8_000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
    });

    if (!res.ok) {
      // FastAPI surfaces validation errors under `detail`
      const body = await res.json().catch(() => ({})) as { detail?: string | { msg: string }[] };
      let message = `HTTP ${res.status}`;
      if (typeof body.detail === 'string') {
        message = body.detail;
      } else if (Array.isArray(body.detail) && body.detail.length > 0) {
        message = body.detail.map((e) => e.msg).join('; ');
      }
      throw new Error(message);
    }

    return res.json() as Promise<T>;
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      throw new Error('Request timed out. Ensure the API server is running.');
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// ── Exported API calls ────────────────────────────────────────────────────

/** GET /health — liveness + readiness probe */
export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

/** GET /models/status — model artefact loading status */
export async function fetchModelsStatus(): Promise<ModelsStatusResponse> {
  return request<ModelsStatusResponse>('/models/status');
}

/** POST /predict — run ensemble prediction */
export async function runPrediction(
  body: PredictionRequest,
): Promise<PredictionResponse> {
  return request<PredictionResponse>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** POST /predict/from-raw - map external API payloads then predict */
export async function runRawDataPrediction(
  body: RawDataPredictionRequest,
): Promise<RawDataPredictionResponse> {
  return request<RawDataPredictionResponse>('/predict/from-raw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** POST /predict/live - lat/lon real-time provider ingestion + prediction */
export async function runLivePrediction(
  body: LivePredictionRequest,
): Promise<LivePredictionResponse> {
  return request<LivePredictionResponse>('/predict/live', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, 15_000);
}

/** GET /history — paginated prediction history */
export async function fetchHistory(
  params: HistoryQueryParams = {},
): Promise<HistoryResponse> {
  const qs = new URLSearchParams();
  if (params.page)          qs.set('page',          String(params.page));
  if (params.page_size)     qs.set('page_size',      String(params.page_size));
  if (params.disaster_type) qs.set('disaster_type',  params.disaster_type);
  if (params.risk_level)    qs.set('risk_level',     params.risk_level);
  if (params.date_from)     qs.set('date_from',      params.date_from);
  if (params.date_to)       qs.set('date_to',        params.date_to);

  const query = qs.toString();
  return request<HistoryResponse>(`/history${query ? `?${query}` : ''}`);
}

/** GET /history/{id} — single prediction record */
export async function fetchHistoryRecord(id: string): Promise<HistoryRecord> {
  return request<HistoryRecord>(`/history/${id}`);
}

/**
 * GET /explainability — Global feature importance and SHAP-style contributions.
 *
 * Returns `available: false` (not an error) when the training pipeline has
 * not yet been run and feature_importance.json does not exist.
 */
export async function fetchExplainability(
  disasterType: ExplainabilityDisasterType,
): Promise<ExplainabilityResponse> {
  const query = new URLSearchParams({ disaster_type: disasterType }).toString();
  return request<ExplainabilityResponse>(`/explainability?${query}`, undefined, 10_000);
}
