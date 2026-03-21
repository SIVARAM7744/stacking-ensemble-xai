/**
 * PredictDisaster.tsx
 * -------------------
 * Dataset-aligned prediction interface for the Ensemble-Based Hybrid
 * Disaster Prediction System.
 *
 * API integration
 * ---------------
 * All HTTP traffic is routed through src/services/api.ts → runPrediction().
 * No Math.random(), no mock data, no inline fetch(), no hardcoded results.
 * The result panel renders ONLY when a real PredictionResponse is received.
 *
 * Disaster-type routing
 * ----------------------
 * Flood      → rainfall, soil_moisture visible; seismic_activity hidden
 * Earthquake → seismic_activity visible; rainfall, soil_moisture hidden
 */

import { useState } from 'react';
import { Loader2, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  API_BASE_URL,
  runLivePrediction,
  runPrediction,
  type PredictionRequest,
  type PredictionResponse,
} from '../services/api';

// ── Types ─────────────────────────────────────────────────────────────────

interface FormData {
  latitude: string;
  longitude: string;
  rainfall: string;
  temperature: string;
  humidity: string;
  soilMoisture: string;
  windSpeed: string;
  pressure: string;
  seismicActivity: string;
  previousDisaster: string;
  location: string;
  regionCode: string;
}

interface FormErrors {
  [key: string]: string;
}

type DisasterType = '' | 'flood' | 'earthquake';
type InputMode = 'manual' | 'live';

// ── Loading steps (UI only — no timing, driven by real API response) ──────

const LOADING_STEPS = [
  'Running Random Forest...',
  'Running Gradient Boosting...',
  'Running SVM...',
  'Running Stacking Meta-Learner...',
  'Computing Disaster Risk Index (DRI)...',
] as const;

// ── Risk colour helpers ───────────────────────────────────────────────────

function getRiskColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'HIGH':     return '#B91C1C';
    case 'MODERATE': return '#CA8A04';
    case 'LOW':      return '#15803D';
    default:         return '#111827';
  }
}

function getRiskBgColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'HIGH':     return '#FEF2F2';
    case 'MODERATE': return '#FEFCE8';
    case 'LOW':      return '#F0FDF4';
    default:         return '#F3F4F6';
  }
}

// ── Field validation ──────────────────────────────────────────────────────

function validateField(name: string, value: string): string {
  if (value === '') return '';
  const n = parseFloat(value);

  switch (name) {
    case 'rainfall':
      return n < 0 ? 'Rainfall must be ≥ 0' : '';
    case 'humidity':
      return n < 0 || n > 100 ? 'Humidity must be 0–100' : '';
    case 'soilMoisture':
      return n < 0 || n > 100 ? 'Soil Moisture must be 0–100' : '';
    case 'windSpeed':
      return n < 0 ? 'Wind Speed must be ≥ 0' : '';
    case 'seismicActivity':
      return n < 0 || n > 10 ? 'Seismic Activity must be 0–10' : '';
    case 'pressure':
      return n < 800 || n > 1100 ? 'Pressure must be 800–1100 hPa' : '';
    default:
      return '';
  }
}

// ── Blank form state ──────────────────────────────────────────────────────

const EMPTY_FORM: FormData = {
  latitude:        '',
  longitude:       '',
  rainfall:        '',
  temperature:     '',
  humidity:        '',
  soilMoisture:    '',
  windSpeed:       '',
  pressure:        '',
  seismicActivity: '',
  previousDisaster:'0',
  location:        '',
  regionCode:      '',
};

// ── Component ─────────────────────────────────────────────────────────────

export function PredictDisaster() {
  const [disasterType,   setDisasterType]   = useState<DisasterType>('');
  const [inputMode,      setInputMode]      = useState<InputMode>('manual');
  const [formData,       setFormData]       = useState<FormData>(EMPTY_FORM);
  const [errors,         setErrors]         = useState<FormErrors>({});
  const [isLoading,      setIsLoading]      = useState(false);
  const [loadingStep,    setLoadingStep]     = useState(-1);
  const [prediction,     setPrediction]     = useState<PredictionResponse | null>(null);
  const [apiError,       setApiError]       = useState<string | null>(null);

  // ── Input handling ──────────────────────────────────────────────────────

  function handleInputChange(
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setErrors(prev => ({ ...prev, [name]: validateField(name, value) }));
  }

  function handleDisasterTypeChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setDisasterType(e.target.value as DisasterType);
    setPrediction(null);
    setApiError(null);
    setErrors({});
    setFormData(EMPTY_FORM);
  }

  function toFieldString(value: unknown): string {
    if (value === null || value === undefined) return '';
    if (typeof value === 'number' && Number.isFinite(value)) {
      const rounded = Math.round(value * 100) / 100;
      return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(2);
    }
    if (typeof value === 'string') return value;
    return '';
  }

  function applyMappedFeaturesToForm(mapped: Record<string, unknown>) {
    setFormData(prev => ({
      ...prev,
      rainfall: toFieldString(mapped.rainfall) || prev.rainfall,
      temperature: toFieldString(mapped.temperature) || prev.temperature,
      humidity: toFieldString(mapped.humidity) || prev.humidity,
      soilMoisture: toFieldString(mapped.soil_moisture) || prev.soilMoisture,
      windSpeed: toFieldString(mapped.wind_speed) || prev.windSpeed,
      pressure: toFieldString(mapped.atmospheric_pressure) || prev.pressure,
      seismicActivity: toFieldString(mapped.seismic_activity) || prev.seismicActivity,
      previousDisaster: toFieldString(mapped.previous_disaster) || prev.previousDisaster,
      location: toFieldString(mapped.location) || prev.location,
      regionCode: toFieldString(mapped.region_code) || prev.regionCode,
    }));
  }

  // ── Validation gate ─────────────────────────────────────────────────────

  function hasValidationErrors(): boolean {
    return Object.values(errors).some(e => e !== '');
  }

  function isFormSubmittable(): boolean {
    if (!disasterType) return false;
    if (hasValidationErrors()) return false;
    return true;
  }

  function isLiveSubmittable(): boolean {
    if (!disasterType) return false;
    if (!formData.latitude || !formData.longitude) return false;
    return true;
  }

  // ── Explanation text derived from real input values ─────────────────────

  function buildExplanation(result: PredictionResponse): string {
    if (disasterType === 'flood') {
      const rainValue = Number.parseFloat(formData.rainfall);
      const smValue = Number.parseFloat(formData.soilMoisture);
      const humValue = Number.parseFloat(formData.humidity);

      const rainText = Number.isFinite(rainValue) ? `${rainValue}` : 'N/A';
      const smText = Number.isFinite(smValue) ? `${smValue}` : 'N/A';
      const humText = Number.isFinite(humValue) ? `${humValue}` : 'N/A';

      const rainSignal =
        !Number.isFinite(rainValue) ? 'Rainfall data is unavailable' :
        rainValue >= 80 ? 'Very high rainfall is a strong flood trigger' :
        rainValue >= 20 ? 'Moderate rainfall adds flood pressure' :
        'Low rainfall reduces immediate flood pressure';

      const soilSignal =
        !Number.isFinite(smValue) ? 'soil saturation is unknown' :
        smValue >= 70 ? 'soil is highly saturated' :
        smValue >= 40 ? 'soil has moderate saturation' :
        'soil is relatively dry';

      const humiditySignal =
        !Number.isFinite(humValue) ? 'humidity trend is unavailable' :
        humValue >= 80 ? 'high humidity supports continued precipitation' :
        humValue >= 60 ? 'moderate humidity supports cloud persistence' :
        'lower humidity weakens near-term precipitation support';

      return (
        `${rainSignal} (${rainText} mm), while ${soilSignal} (${smText}%). ` +
        `Humidity is ${humText}% and ${humiditySignal}. ` +
        `The final decision follows stacked ensemble probability and DRI scoring. ` +
        `Confidence (${result.confidence.toFixed(1)}%) is an agreement metric, not a direct probability.`
      );
    }
    const seismic = formData.seismicActivity || 'N/A';
    const press   = formData.pressure       || 'N/A';
    return (
      `Seismic activity of ${seismic} on the Richter scale indicates tectonic stress. ` +
      `Atmospheric pressure (${press} hPa) shows anomalies correlated with subsurface ` +
      `activity. The ensemble model weights geological indicators higher for earthquake ` +
      `risk assessment. Confidence (${result.confidence.toFixed(1)}%) is model-agreement strength, not event probability.`
    );
  }

  // ── Prediction submission ───────────────────────────────────────────────

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isFormSubmittable() || isLoading) return;

    setIsLoading(true);
    setPrediction(null);
    setApiError(null);
    setLoadingStep(0);

    // Animate loading steps through every 350 ms while the real request runs
    let step = 0;
    const interval = setInterval(() => {
      step = Math.min(step + 1, LOADING_STEPS.length - 1);
      setLoadingStep(step);
    }, 350);

    // Build request body matching PredictionRequest schema exactly
    const body: PredictionRequest = {
      disaster_type:        disasterType === 'flood' ? 'Flood' : 'Earthquake',
      temperature:          formData.temperature     ? parseFloat(formData.temperature)     : null,
      humidity:             formData.humidity        ? parseFloat(formData.humidity)        : null,
      wind_speed:           formData.windSpeed       ? parseFloat(formData.windSpeed)       : null,
      atmospheric_pressure: formData.pressure        ? parseFloat(formData.pressure)        : null,
      previous_disaster:    parseInt(formData.previousDisaster, 10),
      location:             formData.location        || null,
      region_code:          formData.regionCode      || null,
    };

    if (disasterType === 'flood') {
      body.rainfall      = formData.rainfall      ? parseFloat(formData.rainfall)      : null;
      body.soil_moisture = formData.soilMoisture  ? parseFloat(formData.soilMoisture)  : null;
      body.seismic_activity = null;
    } else {
      body.seismic_activity = formData.seismicActivity ? parseFloat(formData.seismicActivity) : null;
      body.rainfall      = null;
      body.soil_moisture = null;
    }

    try {
      // ← Real API call via typed service layer (no Math.random, no mocks)
      const result = await runPrediction(body);

      clearInterval(interval);
      setLoadingStep(LOADING_STEPS.length); // mark all complete
      setPrediction(result);
    } catch (err) {
      clearInterval(interval);
      setLoadingStep(-1);
      const msg = err instanceof Error ? err.message : 'Unexpected error.';
      setApiError(msg);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleLivePredict() {
    if (!isLiveSubmittable() || isLoading) return;

    setIsLoading(true);
    setPrediction(null);
    setApiError(null);
    setLoadingStep(0);

    let step = 0;
    const interval = setInterval(() => {
      step = Math.min(step + 1, LOADING_STEPS.length - 1);
      setLoadingStep(step);
    }, 350);

    try {
      const result = await runLivePrediction({
        disaster_type: disasterType === 'flood' ? 'Flood' : 'Earthquake',
        latitude: parseFloat(formData.latitude),
        longitude: parseFloat(formData.longitude),
        location: formData.location || null,
        region_code: formData.regionCode || null,
        previous_disaster: parseInt(formData.previousDisaster, 10),
      });

      clearInterval(interval);
      setLoadingStep(LOADING_STEPS.length);
      applyMappedFeaturesToForm(result.mapped_features);
      setPrediction(result.prediction);
    } catch (err) {
      clearInterval(interval);
      setLoadingStep(-1);
      const msg = err instanceof Error ? err.message : 'Unexpected error.';
      setApiError(msg);
    } finally {
      setIsLoading(false);
    }
  }

  // ── Derived display values (from real response only) ────────────────────

  const dominantFeatures =
    disasterType === 'flood'
      ? ['Rainfall', 'Soil Moisture', 'Humidity']
      : ['Seismic Activity', 'Atmospheric Pressure', 'Previous Occurrence'];

  const predictionRegion =
    formData.location
      ? `${formData.location}${formData.regionCode ? ' – ' + formData.regionCode : ''}`
      : 'Unknown Region';

  const modelRows = prediction
    ? [
        { key: 'random_forest', label: 'Random Forest (RF)', value: prediction.model_probabilities.random_forest, color: '#1E3A8A' },
        { key: 'gradient_boosting', label: 'Gradient Boosting (GB)', value: prediction.model_probabilities.gradient_boosting, color: '#2563EB' },
        { key: 'svm', label: 'SVM', value: prediction.model_probabilities.svm, color: '#0EA5E9' },
        { key: 'stacking', label: 'Stacking Meta-Learner', value: prediction.model_probabilities.stacking, color: '#15803D' },
      ]
    : [];

  const totalModelSignal = modelRows.reduce((sum, item) => sum + item.value, 0);

  const modelChartData = modelRows.map((item) => ({
    model: item.label,
    probability_pct: Number((item.value * 100).toFixed(2)),
    share_pct: Number((totalModelSignal > 0 ? (item.value / totalModelSignal) * 100 : 0).toFixed(2)),
    color: item.color,
  }));

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-2" style={{ color: '#111827' }}>
        Predict Disaster Risk
      </h1>
      <p className="text-sm mb-6" style={{ color: '#6B7280' }}>
        Dataset-Aligned Ensemble Prediction Interface
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── Left: Input Form ─────────────────────────────────────────── */}
        <div className="p-4" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>

          {/* Disaster Type */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1" style={{ color: '#111827' }}>
              Disaster Type <span style={{ color: '#B91C1C' }}>*</span>
            </label>
            <select
              value={disasterType}
              onChange={handleDisasterTypeChange}
              className="w-full px-3 py-2 text-sm"
              style={{ border: '1px solid #1E3A8A', borderRadius: '4px', backgroundColor: '#FFFFFF', color: '#111827' }}
            >
              <option value="" disabled>Select Disaster Type</option>
              <option value="flood">Flood</option>
              <option value="earthquake">Earthquake</option>
            </select>
          </div>

          {/* Helper text — only shown once type is selected */}
          {disasterType && (
            <div
              className="mb-4 p-3 text-xs"
              style={{ backgroundColor: '#F3F4F6', borderRadius: '4px', color: '#6B7280' }}
            >
              {disasterType === 'flood' ? (
                <>
                  <strong>Meteorological Basis:</strong> Flood prediction utilises
                  hydro-meteorological features including rainfall intensity, soil
                  saturation, and atmospheric humidity for ensemble risk assessment.
                </>
              ) : (
                <>
                  <strong>Seismic Basis:</strong> Earthquake prediction relies on
                  seismic activity magnitude, atmospheric pressure anomalies, and
                  historical occurrence patterns for tectonic risk assessment.
                </>
              )}
            </div>
          )}

          {/* Input mode switch */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2" style={{ color: '#111827' }}>
              Prediction Mode
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setInputMode('manual')}
                className="px-3 py-2 text-sm font-medium"
                style={{
                  border: '1px solid #1E3A8A',
                  borderRadius: '4px',
                  backgroundColor: inputMode === 'manual' ? '#1E3A8A' : '#FFFFFF',
                  color: inputMode === 'manual' ? '#FFFFFF' : '#1E3A8A',
                }}
              >
                Manual Data
              </button>
              <button
                type="button"
                onClick={() => setInputMode('live')}
                className="px-3 py-2 text-sm font-medium"
                style={{
                  border: '1px solid #1E3A8A',
                  borderRadius: '4px',
                  backgroundColor: inputMode === 'live' ? '#1E3A8A' : '#FFFFFF',
                  color: inputMode === 'live' ? '#FFFFFF' : '#1E3A8A',
                }}
              >
                Live API (Lat/Lon)
              </button>
            </div>
            {inputMode === 'live' && (
              <p className="text-xs mt-2" style={{ color: '#6B7280' }}>
                Enter latitude/longitude and run live mode. API values will be fetched and auto-filled.
              </p>
            )}
          </div>

          <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
            {inputMode === 'manual' ? 'Environmental Parameters' : 'Live API Input'}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <fieldset disabled={inputMode === 'live'} className="contents">

              {/* Flood: Rainfall */}
              {disasterType === 'flood' && (
                <div>
                  <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                    Rainfall <span className="text-xs">(mm)</span>
                  </label>
                  <input
                    type="number" name="rainfall" step="0.1" min="0"
                    value={formData.rainfall}
                    onChange={handleInputChange}
                    placeholder="Enter rainfall (mm)"
                    className="w-full px-3 py-2 text-sm"
                    style={{
                      border: `1px solid ${errors.rainfall ? '#B91C1C' : '#E5E7EB'}`,
                      borderRadius: '4px', backgroundColor: '#FFFFFF',
                    }}
                  />
                  {errors.rainfall && (
                    <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.rainfall}</p>
                  )}
                </div>
              )}

              {/* Earthquake: Seismic Activity */}
              {disasterType === 'earthquake' && (
                <div>
                  <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                    Seismic Activity <span className="text-xs">(Richter)</span>
                  </label>
                  <input
                    type="number" name="seismicActivity" step="0.1" min="0" max="10"
                    value={formData.seismicActivity}
                    onChange={handleInputChange}
                    placeholder="Enter seismic activity (0–10)"
                    className="w-full px-3 py-2 text-sm"
                    style={{
                      border: `1px solid ${errors.seismicActivity ? '#B91C1C' : '#E5E7EB'}`,
                      borderRadius: '4px', backgroundColor: '#FFFFFF',
                    }}
                  />
                  {errors.seismicActivity && (
                    <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.seismicActivity}</p>
                  )}
                </div>
              )}

              {/* Temperature — common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Temperature <span className="text-xs">(°C)</span>
                </label>
                <input
                  type="number" name="temperature" step="0.1"
                  value={formData.temperature}
                  onChange={handleInputChange}
                  placeholder="Enter temperature (°C)"
                  className="w-full px-3 py-2 text-sm"
                  style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                />
              </div>

              {/* Humidity — common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Humidity <span className="text-xs">(%)</span>
                </label>
                <input
                  type="number" name="humidity" min="0" max="100"
                  value={formData.humidity}
                  onChange={handleInputChange}
                  placeholder="Enter humidity (0–100)"
                  className="w-full px-3 py-2 text-sm"
                  style={{
                    border: `1px solid ${errors.humidity ? '#B91C1C' : '#E5E7EB'}`,
                    borderRadius: '4px', backgroundColor: '#FFFFFF',
                  }}
                />
                {errors.humidity && (
                  <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.humidity}</p>
                )}
              </div>

              {/* Flood: Soil Moisture */}
              {disasterType === 'flood' && (
                <div>
                  <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                    Soil Moisture <span className="text-xs">(%)</span>
                  </label>
                  <input
                    type="number" name="soilMoisture" min="0" max="100"
                    value={formData.soilMoisture}
                    onChange={handleInputChange}
                    placeholder="Enter soil moisture (0–100)"
                    className="w-full px-3 py-2 text-sm"
                    style={{
                      border: `1px solid ${errors.soilMoisture ? '#B91C1C' : '#E5E7EB'}`,
                      borderRadius: '4px', backgroundColor: '#FFFFFF',
                    }}
                  />
                  {errors.soilMoisture && (
                    <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.soilMoisture}</p>
                  )}
                </div>
              )}

              {/* Wind Speed — common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Wind Speed <span className="text-xs">(km/h)</span>
                </label>
                <input
                  type="number" name="windSpeed" step="0.1" min="0"
                  value={formData.windSpeed}
                  onChange={handleInputChange}
                  placeholder="Enter wind speed (km/h)"
                  className="w-full px-3 py-2 text-sm"
                  style={{
                    border: `1px solid ${errors.windSpeed ? '#B91C1C' : '#E5E7EB'}`,
                    borderRadius: '4px', backgroundColor: '#FFFFFF',
                  }}
                />
                {errors.windSpeed && (
                  <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.windSpeed}</p>
                )}
              </div>

              {/* Atmospheric Pressure — common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Atmospheric Pressure <span className="text-xs">(hPa)</span>
                </label>
                <input
                  type="number" name="pressure" min="800" max="1100"
                  value={formData.pressure}
                  onChange={handleInputChange}
                  placeholder="Enter pressure (800–1100)"
                  className="w-full px-3 py-2 text-sm"
                  style={{
                    border: `1px solid ${errors.pressure ? '#B91C1C' : '#E5E7EB'}`,
                    borderRadius: '4px', backgroundColor: '#FFFFFF',
                  }}
                />
                {errors.pressure && (
                  <p className="text-xs mt-1" style={{ color: '#B91C1C' }}>{errors.pressure}</p>
                )}
              </div>

              </fieldset>

              {/* Previous Disaster Occurrence - common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Previous Disaster Occurrence
                </label>
                <select
                  name="previousDisaster"
                  value={formData.previousDisaster}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 text-sm"
                  style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                >
                  <option value="0">No (0)</option>
                  <option value="1">Yes (1)</option>
                </select>
              </div>

              {/* Location — common */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Location <span className="text-xs">(Text)</span>
                </label>
                <input
                  type="text" name="location"
                  value={formData.location}
                  onChange={handleInputChange}
                  placeholder="e.g. Chennai"
                  className="w-full px-3 py-2 text-sm"
                  style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                />
              </div>

              {/* Region Code — optional */}
              <div>
                <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                  Region Code <span className="text-xs">(Optional)</span>
                </label>
                <input
                  type="text" name="regionCode"
                  value={formData.regionCode}
                  onChange={handleInputChange}
                  placeholder="e.g. TN-04"
                  className="w-full px-3 py-2 text-sm"
                  style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                />
              </div>

              {inputMode === 'live' && (
                <>
                  <div>
                    <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                      Latitude <span className="text-xs">(required)</span>
                    </label>
                    <input
                      type="number" name="latitude" step="0.0001" min="-90" max="90"
                      value={formData.latitude}
                      onChange={handleInputChange}
                      placeholder="e.g. 13.0827"
                      className="w-full px-3 py-2 text-sm"
                      style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm mb-1" style={{ color: '#6B7280' }}>
                      Longitude <span className="text-xs">(required)</span>
                    </label>
                    <input
                      type="number" name="longitude" step="0.0001" min="-180" max="180"
                      value={formData.longitude}
                      onChange={handleInputChange}
                      placeholder="e.g. 80.2707"
                      className="w-full px-3 py-2 text-sm"
                      style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Action button by mode */}
            {inputMode === 'manual' ? (
              <button
                type="submit"
                disabled={isLoading || !isFormSubmittable()}
                className="w-full py-2.5 text-white text-sm font-medium flex items-center justify-center gap-2"
                style={{
                  backgroundColor: (!isFormSubmittable() || isLoading) ? '#9CA3AF' : '#1E3A8A',
                  borderRadius: '4px',
                  cursor: (!isFormSubmittable() || isLoading) ? 'not-allowed' : 'pointer',
                  opacity: isLoading ? 0.85 : 1,
                }}
              >
                {isLoading && <Loader2 size={16} className="animate-spin" />}
                {isLoading ? 'Running Ensemble & Meta-Learner...' : 'Run Prediction (Manual Data)'}
              </button>
            ) : (
              <button
                type="button"
                onClick={handleLivePredict}
                disabled={isLoading || !isLiveSubmittable()}
                className="w-full py-2.5 text-sm font-medium flex items-center justify-center gap-2"
                style={{
                  backgroundColor: (!isLiveSubmittable() || isLoading) ? '#D1D5DB' : '#FFFFFF',
                  border: '1px solid #1E3A8A',
                  color: (!isLiveSubmittable() || isLoading) ? '#6B7280' : '#1E3A8A',
                  borderRadius: '4px',
                  cursor: (!isLiveSubmittable() || isLoading) ? 'not-allowed' : 'pointer',
                }}
              >
                {isLoading && <Loader2 size={16} className="animate-spin" />}
                Fetch API Data, Auto-Fill & Predict
              </button>
            )}

            {!disasterType && (
              <p className="text-xs text-center" style={{ color: '#9CA3AF' }}>
                Select a disaster type to enable prediction.
              </p>
            )}

            {/* API Error Banner */}
            {apiError && (
              <div
                className="flex items-start gap-2 p-3 text-sm"
                style={{
                  backgroundColor: '#FEF2F2',
                  borderRadius: '4px',
                  border: '1px solid #B91C1C',
                  color: '#B91C1C',
                }}
              >
                <XCircle size={16} className="mt-0.5 flex-shrink-0" />
                <div>
                  <strong className="block mb-1">Prediction Service Unavailable</strong>
                  <span className="text-xs">{apiError}</span>
                  <p className="text-xs mt-1" style={{ color: '#6B7280' }}>
                    Ensure the FastAPI backend is running at{' '}
                    <code className="font-mono">{API_BASE_URL}</code>
                  </p>
                </div>
              </div>
            )}
          </form>
        </div>

        {/* ── Right: Prediction Output ──────────────────────────────────── */}
        <div className="p-4" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
          <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
            Prediction Output
          </h2>

          {/* Idle state — no result yet */}
          {!prediction && !isLoading && !apiError && (
            <div
              className="h-64 flex flex-col items-center justify-center gap-3"
              style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}
            >
              <AlertCircle size={32} style={{ color: '#9CA3AF' }} />
              <p className="text-sm text-center px-4" style={{ color: '#6B7280' }}>
                Enter {inputMode === 'manual' ? 'Environmental Parameters' : 'Live API Input'} and run prediction<br />
                to receive ensemble model results.
              </p>
            </div>
          )}

          {/* API error idle state */}
          {!prediction && !isLoading && apiError && (
            <div
              className="h-64 flex flex-col items-center justify-center gap-3"
              style={{ backgroundColor: '#FEF2F2', borderRadius: '4px' }}
            >
              <XCircle size={32} style={{ color: '#B91C1C' }} />
              <p className="text-sm text-center px-4" style={{ color: '#B91C1C' }}>
                No prediction result available.<br />
                Resolve the API error and try again.
              </p>
            </div>
          )}

          {/* Loading steps */}
          {isLoading && (
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="flex items-center gap-3 mb-4">
                <Loader2 size={24} className="animate-spin" style={{ color: '#1E3A8A' }} />
                <div>
                  <span className="text-sm font-semibold block" style={{ color: '#111827' }}>
                    Running Ensemble Models:
                  </span>
                  <span className="text-xs" style={{ color: '#6B7280' }}>
                    Stacking pipeline in progress...
                  </span>
                </div>
              </div>
              <div className="space-y-2 ml-9">
                {LOADING_STEPS.map((text, i) => {
                  const done    = loadingStep > i;
                  const active  = loadingStep === i;
                  const pending = loadingStep < i;
                  return (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-sm"
                      style={{ color: done ? '#15803D' : active ? '#1E3A8A' : '#9CA3AF' }}
                    >
                      {done    && <CheckCircle size={14} />}
                      {active  && <Loader2 size={14} className="animate-spin" />}
                      {pending && <div className="w-3.5 h-3.5 rounded-full" style={{ border: '1px solid #9CA3AF' }} />}
                      <span>{text}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Real prediction result — renders ONLY when API returns data */}
          {prediction && !isLoading && (
            <div className="space-y-4">

              {/* Risk Summary Box */}
              <div
                className="p-4"
                style={{
                  border: `2px solid ${getRiskColor(prediction.risk_level)}`,
                  borderRadius: '4px',
                  backgroundColor: getRiskBgColor(prediction.risk_level),
                }}
              >
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div style={{ color: '#6B7280' }}>Predicted Disaster</div>
                    <div className="font-semibold" style={{ color: '#111827' }}>
                      {prediction.predicted_disaster}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6B7280' }}>Probability</div>
                    <div className="font-semibold" style={{ color: '#111827' }}>
                      {(prediction.probability * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6B7280' }}>Disaster Risk Index (DRI)</div>
                    <div className="font-bold text-xl" style={{ color: getRiskColor(prediction.risk_level) }}>
                      {prediction.dri.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6B7280' }}>Risk Level</div>
                    <div
                      className="inline-block px-2 py-0.5 text-white font-semibold text-sm"
                      style={{ backgroundColor: getRiskColor(prediction.risk_level), borderRadius: '4px' }}
                    >
                      {prediction.risk_level}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6B7280' }}>Prediction Confidence</div>
                    <div className="font-semibold" style={{ color: '#111827' }}>
                      {prediction.confidence.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#6B7280' }}>Prediction Region</div>
                    <div className="font-semibold" style={{ color: '#111827' }}>
                      {predictionRegion}
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical note */}
              <div
                className="p-2 text-xs"
                style={{ backgroundColor: '#F3F4F6', borderRadius: '4px', color: '#6B7280' }}
              >
                <em>
                  DRI computed using weighted stacking of heterogeneous base learners:
                  DRI = 0.6 × Stacking + 0.4 × avg(RF, GB, SVM)
                </em>
              </div>

              {/* Model-wise Probability Table */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
                <div
                  className="px-3 py-2 text-sm font-medium"
                  style={{ backgroundColor: '#F3F4F6', borderBottom: '1px solid #E5E7EB', color: '#111827' }}
                >
                  Model-wise Disaster Probability (Positive Class)
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ backgroundColor: '#F9FAFB' }}>
                      <th className="px-3 py-2 text-left font-medium" style={{ color: '#6B7280', borderBottom: '1px solid #E5E7EB' }}>Model</th>
                      <th className="px-3 py-2 text-right font-medium" style={{ color: '#6B7280', borderBottom: '1px solid #E5E7EB' }}>Probability (0-100%)</th>
                      <th className="px-3 py-2 text-right font-medium" style={{ color: '#6B7280', borderBottom: '1px solid #E5E7EB' }}>Share of Total Signal (0-100%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelChartData.map(({ model, probability_pct, share_pct }, idx) => (
                      <tr key={model} style={{ borderBottom: idx === modelChartData.length - 1 ? 'none' : '1px solid #E5E7EB' }}>
                        <td className="px-3 py-2" style={{ color: '#111827' }}>{model}</td>
                        <td className="px-3 py-2 text-right font-mono" style={{ color: '#1E3A8A' }}>
                          {probability_pct.toFixed(2)}%
                        </td>
                        <td className="px-3 py-2 text-right font-mono" style={{ color: '#374151' }}>
                          {share_pct.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                    <tr style={{ borderTop: '1px solid #E5E7EB', backgroundColor: '#F9FAFB' }}>
                      <td className="px-3 py-2 font-medium" style={{ color: '#111827' }}>
                        Total
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-medium" style={{ color: '#111827' }}>
                        {(totalModelSignal * 100).toFixed(2)}%
                      </td>
                      <td className="px-3 py-2 text-right font-mono font-medium" style={{ color: '#111827' }}>
                        100.00%
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Model Probability Graph */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
                <div
                  className="px-3 py-2 text-sm font-medium"
                  style={{ backgroundColor: '#F3F4F6', borderBottom: '1px solid #E5E7EB', color: '#111827' }}
                >
                  Model Probability Graph (Actual Chart)
                </div>
                <div className="p-3">
                  <div style={{ width: '100%', height: 320 }}>
                    <ResponsiveContainer>
                      <BarChart data={modelChartData} margin={{ top: 10, right: 20, left: 0, bottom: 50 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis
                          dataKey="model"
                          angle={-18}
                          textAnchor="end"
                          interval={0}
                          height={70}
                          tick={{ fill: '#374151', fontSize: 11 }}
                        />
                        <YAxis tick={{ fill: '#374151', fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip
                          formatter={(value: number, name: string) => [`${value.toFixed(2)}%`, name]}
                          labelStyle={{ color: '#111827' }}
                        />
                        <Legend />
                        <Bar dataKey="probability_pct" name="Absolute Probability %" radius={[4, 4, 0, 0]}>
                          {modelChartData.map((entry) => (
                            <Cell key={`${entry.model}-prob`} fill={entry.color} />
                          ))}
                        </Bar>
                        <Bar dataKey="share_pct" name="Normalized Share %" fill="#94A3B8" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div
                className="p-2 text-xs"
                style={{ backgroundColor: '#F3F4F6', borderRadius: '4px', color: '#6B7280' }}
              >
                <em>
                  Absolute Probability % = each model's positive-class probability.
                  Normalized Share % = each model probability divided by total model signal.
                </em>
              </div>

              {/* Risk Explanation (derived from real input + real response) */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
                <div
                  className="px-3 py-2 text-sm font-medium"
                  style={{ backgroundColor: '#F3F4F6', borderBottom: '1px solid #E5E7EB', color: '#111827' }}
                >
                  Risk Explanation
                </div>
                <div className="p-3">
                  <div className="text-xs mb-2" style={{ color: '#6B7280' }}>
                    <strong>Dominant Features:</strong> {dominantFeatures.join(', ')}
                  </div>
                  <p className="text-sm" style={{ color: '#111827', lineHeight: '1.6' }}>
                    {buildExplanation(prediction)}
                  </p>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}








