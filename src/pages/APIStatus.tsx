import { useState, useEffect, useCallback } from 'react';
import { CheckCircle, XCircle, RefreshCw, Loader2, WifiOff } from 'lucide-react';
import { API_BASE_URL, fetchHealth, fetchModelsStatus } from '../services/api';

interface HealthData {
  status: string;
  api_version: string;
  db_connected: boolean;
  db_engine: string;
}

interface ModelsData {
  status: string;
  models: string[];
  message: string;
}

export function APIStatus() {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [modelsData, setModelsData] = useState<ModelsData | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    setIsRefreshing(true);
    setFetchError(null);

    try {
      const health = await fetchHealth();
      setHealthData(health);
      setBackendOnline(true);

      try {
        const models = await fetchModelsStatus();
        setModelsData(models);
      } catch {
        setModelsData(null);
      }
    } catch {
      setBackendOnline(false);
      setHealthData(null);
      setModelsData(null);
      setFetchError('Unable to reach backend. Ensure the API server is running.');
    } finally {
      setIsRefreshing(false);
      setLastRefresh(new Date());
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const modelDisplayNames: Record<string, string> = {
    Flood: 'Flood Model Set',
    Earthquake: 'Earthquake Model Set',
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: '#111827' }}>
            API Status & System Health
          </h1>
          <p className="text-sm" style={{ color: '#6B7280' }}>
            Live monitoring of prediction API endpoints and infrastructure
          </p>
        </div>
        <button
          onClick={fetchStatus}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-4 py-2 text-sm text-white"
          style={{
            backgroundColor: '#1E3A8A',
            borderRadius: '4px',
            opacity: isRefreshing ? 0.7 : 1,
            cursor: isRefreshing ? 'not-allowed' : 'pointer',
          }}
        >
          {isRefreshing ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
          Refresh
        </button>
      </div>

      {lastRefresh && (
        <p className="text-xs mb-4" style={{ color: '#9CA3AF' }}>
          Last checked: {lastRefresh.toLocaleTimeString()}
        </p>
      )}

      {backendOnline === null && isRefreshing && (
        <div
          className="p-4 mb-6 flex items-center gap-3"
          style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#F9FAFB' }}
        >
          <Loader2 size={20} className="animate-spin" style={{ color: '#6B7280' }} />
          <span className="text-sm" style={{ color: '#6B7280' }}>
            Checking backend status...
          </span>
        </div>
      )}

      {backendOnline === true && healthData && (
        <div
          className="p-4 mb-6 flex items-center gap-4"
          style={{ border: '2px solid #15803D', borderRadius: '4px', backgroundColor: '#F0FDF4' }}
        >
          <CheckCircle size={28} style={{ color: '#15803D' }} />
          <div>
            <div className="font-semibold" style={{ color: '#15803D' }}>
              Backend Online
            </div>
            <div className="text-sm" style={{ color: '#166534' }}>
              API status: {healthData.status} | version {healthData.api_version}
            </div>
          </div>
        </div>
      )}

      {backendOnline === false && (
        <div
          className="p-4 mb-6 flex items-center gap-4"
          style={{ border: '2px solid #B91C1C', borderRadius: '4px', backgroundColor: '#FEF2F2' }}
        >
          <WifiOff size={28} style={{ color: '#B91C1C' }} />
          <div>
            <div className="font-semibold" style={{ color: '#B91C1C' }}>
              Backend Offline
            </div>
            <div className="text-sm" style={{ color: '#991B1B' }}>
              {fetchError ?? 'No API telemetry data available.'}
            </div>
          </div>
        </div>
      )}

      <div className="p-4 mb-6" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
          System Metrics
        </h2>

        {backendOnline === true && healthData ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>API Status</div>
              <div className="text-base font-semibold" style={{ color: '#15803D' }}>{healthData.status}</div>
            </div>
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>API Version</div>
              <div className="text-base font-semibold" style={{ color: '#1E3A8A' }}>{healthData.api_version}</div>
            </div>
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>Deployment Framework</div>
              <div className="text-base font-semibold" style={{ color: '#1E3A8A' }}>FastAPI</div>
            </div>
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>Average Latency</div>
              <div className="text-base font-semibold" style={{ color: '#6B7280' }}>-</div>
            </div>
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>Total Predictions Served</div>
              <div className="text-base font-semibold" style={{ color: '#6B7280' }}>-</div>
            </div>
            <div className="p-4" style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}>
              <div className="text-xs mb-1" style={{ color: '#6B7280' }}>Database</div>
              <div className="text-base font-semibold" style={{ color: healthData.db_connected ? '#15803D' : '#B91C1C' }}>
                {healthData.db_connected
                  ? `${healthData.db_engine} Connected`
                  : `${healthData.db_engine} Not Connected`}
              </div>
            </div>
          </div>
        ) : (
          <div
            className="p-10 flex flex-col items-center text-center"
            style={{ backgroundColor: '#F9FAFB', borderRadius: '4px', border: '1px solid #E5E7EB' }}
          >
            <XCircle size={32} style={{ color: '#D1D5DB' }} className="mb-3" />
            <p className="text-sm font-medium" style={{ color: '#374151' }}>
              No API telemetry data available.
            </p>
          </div>
        )}
      </div>

      <div className="p-4 mb-6" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
          Registered API Endpoints
        </h2>

        {backendOnline === true ? (
          <div className="space-y-2">
            {[
              { name: 'GET /', description: 'Root system info' },
              { name: 'GET /health', description: 'Health check' },
              { name: 'POST /predict', description: 'Ensemble disaster prediction' },
              { name: 'POST /predict/from-raw', description: 'Provider payload mapping + prediction' },
              { name: 'POST /predict/live', description: 'Live lat/lon provider fetch + prediction' },
              { name: 'GET /models/status', description: 'Model loading status' },
              { name: 'GET /history', description: 'Prediction history' },
              { name: 'GET /explainability', description: 'Feature importance and SHAP-style output' },
            ].map((ep) => (
              <div
                key={ep.name}
                className="p-3 flex items-center justify-between"
                style={{ backgroundColor: '#F3F4F6', borderRadius: '4px' }}
              >
                <div className="flex items-center gap-3">
                  <CheckCircle size={15} style={{ color: '#15803D' }} />
                  <div>
                    <div className="font-mono text-sm font-medium" style={{ color: '#111827' }}>
                      {ep.name}
                    </div>
                    <div className="text-xs" style={{ color: '#6B7280' }}>{ep.description}</div>
                  </div>
                </div>
                <span
                  className="text-xs px-2 py-0.5"
                  style={{ backgroundColor: '#15803D', color: '#FFFFFF', borderRadius: '4px' }}
                >
                  Active
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center" style={{ backgroundColor: '#F9FAFB', borderRadius: '4px', border: '1px solid #E5E7EB' }}>
            <p className="text-sm" style={{ color: '#6B7280' }}>
              Endpoint status unavailable because backend is offline.
            </p>
          </div>
        )}
      </div>

      <div className="p-4" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
        <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
          Loaded Model Sets
        </h2>

        {backendOnline === true && modelsData ? (
          <>
            <div
              className="mb-3 p-3 text-sm"
              style={{
                backgroundColor: modelsData.status === 'loaded' ? '#F0FDF4' : '#FEF2F2',
                border: `1px solid ${modelsData.status === 'loaded' ? '#15803D' : '#B91C1C'}`,
                borderRadius: '4px',
                color: modelsData.status === 'loaded' ? '#15803D' : '#B91C1C',
              }}
            >
              {modelsData.message}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ backgroundColor: '#1E3A8A' }}>
                    <th className="text-left p-3 text-white font-medium">Model</th>
                    <th className="text-left p-3 text-white font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(modelsData.models ?? []).map((modelKey, index) => (
                    <tr
                      key={modelKey}
                      style={{
                        backgroundColor: index % 2 === 0 ? '#FFFFFF' : '#F3F4F6',
                        borderBottom: '1px solid #E5E7EB',
                      }}
                    >
                      <td className="p-3 font-medium" style={{ color: '#111827' }}>
                        {modelDisplayNames[modelKey] ?? modelKey}
                      </td>
                      <td className="p-3">
                        <span className="flex items-center gap-2 text-sm" style={{ color: '#15803D' }}>
                          <CheckCircle size={14} /> Loaded
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="p-10 text-center" style={{ backgroundColor: '#F9FAFB', borderRadius: '4px', border: '1px solid #E5E7EB' }}>
            <XCircle size={32} style={{ color: '#D1D5DB' }} className="mb-3 mx-auto" />
            <p className="text-sm font-medium" style={{ color: '#374151' }}>
              Model status unknown.
            </p>
            <p className="text-xs mt-1" style={{ color: '#6B7280' }}>
              Run the training pipeline and start the backend to see model status.
            </p>
          </div>
        )}
      </div>

      <p className="text-xs mt-4" style={{ color: '#9CA3AF' }}>
        API base URL: <code>{API_BASE_URL}</code>
      </p>
    </div>
  );
}
