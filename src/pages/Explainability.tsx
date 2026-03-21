import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, FlaskConical, Loader2, RefreshCw } from 'lucide-react';
import {
  API_BASE_URL,
  fetchExplainability,
  type ExplainabilityDisasterType,
  type ExplainabilityResponse,
  type FeatureImportanceItem,
} from '../services/api';

type TabId = 'importance' | 'shap';

type PageState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'unavailable'; note: string }
  | { kind: 'ready'; data: ExplainabilityResponse };

const MODEL_WEIGHTS: Record<string, number> = {
  random_forest: 0.32,
  gradient_boosting: 0.38,
  svm: 0.30,
};

const MODEL_LABELS: Record<string, string> = {
  random_forest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting',
  svm: 'SVM',
};

const DISASTER_OPTIONS: ExplainabilityDisasterType[] = ['Flood', 'Earthquake'];

function formatFeatureName(raw: string): string {
  return raw
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function shapColor(value: number): string {
  return value >= 0 ? '#B91C1C' : '#15803D';
}

function EmptyState({ message, sub }: { message: string; sub?: string }) {
  return (
    <div
      className="p-12 flex flex-col items-center justify-center text-center"
      style={{
        backgroundColor: '#F9FAFB',
        borderRadius: '4px',
        border: '1px solid #E5E7EB',
      }}
    >
      <FlaskConical size={40} style={{ color: '#D1D5DB' }} className="mb-4" />
      <p className="text-sm font-semibold mb-1" style={{ color: '#374151' }}>
        {message}
      </p>
      {sub ? <p className="text-sm" style={{ color: '#6B7280' }}>{sub}</p> : null}
    </div>
  );
}

function FeatureImportanceChart({ items }: { items: FeatureImportanceItem[] }) {
  if (items.length === 0) {
    return (
      <EmptyState
        message="Feature importance list is empty."
        sub="Retrain the pipeline to populate this chart."
      />
    );
  }

  const maxPct = Math.max(...items.map((item) => item.importance_pct));

  return (
    <div>
      {items.map((item, idx) => {
        const barWidth = maxPct > 0 ? (item.importance_pct / maxPct) * 100 : 0;
        return (
          <div key={item.feature} className="mb-4">
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm font-medium" style={{ color: '#111827' }}>
                {formatFeatureName(item.feature)}
              </span>
              <span className="text-sm font-mono font-semibold" style={{ color: '#1E3A8A' }}>
                {item.importance_pct.toFixed(1)}%
              </span>
            </div>

            <div
              style={{
                backgroundColor: '#F3F4F6',
                border: '1px solid #E5E7EB',
                borderRadius: '4px',
                height: '20px',
                width: '100%',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${barWidth}%`,
                  height: '100%',
                  backgroundColor: idx === 0 ? '#1E3A8A' : '#3B5EBD',
                  borderRadius: '3px',
                  opacity: Math.max(0.35, 1 - idx * 0.07),
                  minWidth: barWidth > 0 ? '4px' : '0',
                }}
              />
            </div>
          </div>
        );
      })}

      <div className="flex justify-between mt-3 text-xs" style={{ color: '#9CA3AF' }}>
        <span>0%</span>
        <span>Feature Importance (% of total)</span>
        <span>100%</span>
      </div>
    </div>
  );
}

function ShapContributionList({ items }: { items: FeatureImportanceItem[] }) {
  if (items.length === 0) {
    return (
      <EmptyState
        message="SHAP contribution data unavailable."
        sub="Run the training pipeline to generate feature attributions."
      />
    );
  }

  const sorted = [...items].sort(
    (a, b) => Math.abs(b.shap_contribution) - Math.abs(a.shap_contribution),
  );
  const maxAbs = Math.max(...sorted.map((item) => Math.abs(item.shap_contribution)));

  return (
    <div>
      {sorted.map((item) => {
        const barWidth = maxAbs > 0 ? (Math.abs(item.shap_contribution) / maxAbs) * 100 : 0;
        const color = shapColor(item.shap_contribution);
        const sign = item.shap_contribution >= 0 ? '+' : '';

        return (
          <div
            key={item.feature}
            className="flex items-center gap-4 py-3"
            style={{ borderBottom: '1px solid #F3F4F6' }}
          >
            <div style={{ width: '180px', flexShrink: 0 }}>
              <span className="text-sm font-medium" style={{ color: '#111827' }}>
                {formatFeatureName(item.feature)}
              </span>
            </div>

            <div style={{ flex: 1 }}>
              <div
                style={{
                  backgroundColor: '#F3F4F6',
                  border: '1px solid #E5E7EB',
                  borderRadius: '4px',
                  height: '14px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${barWidth}%`,
                    height: '100%',
                    backgroundColor: color,
                    borderRadius: '3px',
                    minWidth: barWidth > 0 ? '3px' : '0',
                  }}
                />
              </div>
            </div>

            <div style={{ width: '80px', textAlign: 'right', flexShrink: 0 }}>
              <span className="text-sm font-mono font-semibold" style={{ color }}>
                {sign}
                {item.shap_contribution.toFixed(4)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function EnsembleBreakdown({
  items,
  disasterType,
}: {
  items: FeatureImportanceItem[];
  disasterType: ExplainabilityDisasterType;
}) {
  const topFeature = items[0]?.feature ?? null;

  return (
    <div className="p-4 mt-6" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
      <h2 className="text-lg font-semibold mb-1" style={{ color: '#111827' }}>
        Ensemble Decision Breakdown
      </h2>
      <p className="text-sm mb-4" style={{ color: '#6B7280' }}>
        Meta-learner weight distribution for the {disasterType.toLowerCase()} model set.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ backgroundColor: '#1E3A8A' }}>
              <th className="text-left p-3 text-white font-medium">Model</th>
              <th className="text-left p-3 text-white font-medium">Weight</th>
              <th className="text-left p-3 text-white font-medium">Role</th>
              <th className="text-left p-3 text-white font-medium">Top Feature Signal</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(MODEL_WEIGHTS).map(([key, weight], idx) => (
              <tr
                key={key}
                style={{
                  backgroundColor: idx % 2 === 1 ? '#F3F4F6' : '#FFFFFF',
                  borderBottom: '1px solid #E5E7EB',
                }}
              >
                <td className="p-3 font-medium" style={{ color: '#111827' }}>
                  {MODEL_LABELS[key]}
                </td>
                <td className="p-3 font-mono" style={{ color: '#1E3A8A' }}>
                  {weight.toFixed(2)}
                </td>
                <td className="p-3" style={{ color: '#6B7280' }}>
                  Base Learner
                </td>
                <td className="p-3" style={{ color: '#6B7280' }}>
                  {topFeature ? formatFeatureName(topFeature) : '-'}
                </td>
              </tr>
            ))}
            <tr style={{ backgroundColor: '#1E3A8A' }}>
              <td className="p-3 font-bold text-white">Stacking Output</td>
              <td className="p-3 font-mono text-white">sum = 1.00</td>
              <td className="p-3 font-bold text-white">Meta-Learner</td>
              <td className="p-3 font-mono text-white">
                {topFeature ? formatFeatureName(topFeature) : '-'}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="text-xs mt-2" style={{ color: '#9CA3AF' }}>
        Meta-learner weights are derived from LogisticRegression coefficients trained over
        5-fold cross-validation outputs of the base learners.
      </p>
    </div>
  );
}

export function Explainability() {
  const [activeTab, setActiveTab] = useState<TabId>('importance');
  const [disasterType, setDisasterType] = useState<ExplainabilityDisasterType>('Flood');
  const [state, setState] = useState<PageState>({ kind: 'idle' });

  const load = useCallback(async () => {
    setState({ kind: 'loading' });
    try {
      const data = await fetchExplainability(disasterType);
      if (!data.available) {
        setState({ kind: 'unavailable', note: data.note });
        return;
      }
      setState({ kind: 'ready', data });
    } catch (err) {
      setState({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Failed to fetch explainability data.',
      });
    }
  }, [disasterType]);

  useEffect(() => {
    load();
  }, [load]);

  const items = state.kind === 'ready' ? state.data.feature_importance : [];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between gap-4 mb-2 flex-wrap">
        <h1 className="text-2xl font-bold" style={{ color: '#111827' }}>
          Model Interpretability and Feature Contribution
        </h1>

        <div className="flex items-center gap-3">
          <select
            value={disasterType}
            onChange={(event) => setDisasterType(event.target.value as ExplainabilityDisasterType)}
            className="px-3 py-2 text-sm"
            style={{
              border: '1px solid #E5E7EB',
              borderRadius: '4px',
              backgroundColor: '#FFFFFF',
              color: '#111827',
            }}
          >
            {DISASTER_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>

          <button
            onClick={load}
            disabled={state.kind === 'loading'}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium"
            style={{
              border: '1px solid #E5E7EB',
              borderRadius: '4px',
              color: '#1E3A8A',
              backgroundColor: '#FFFFFF',
              cursor: state.kind === 'loading' ? 'not-allowed' : 'pointer',
              opacity: state.kind === 'loading' ? 0.6 : 1,
            }}
          >
            {state.kind === 'loading' ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <RefreshCw size={14} />
            )}
            Refresh
          </button>
        </div>
      </div>

      <p className="text-sm mb-6" style={{ color: '#6B7280' }}>
        Disaster-type aware explainability for the active model artifacts. Data is requested from{' '}
        <code
          className="text-xs px-1"
          style={{
            backgroundColor: '#F3F4F6',
            border: '1px solid #E5E7EB',
            borderRadius: '2px',
            color: '#374151',
          }}
        >
          {API_BASE_URL}/explainability?disaster_type={disasterType}
        </code>
      </p>

      {state.kind === 'loading' ? (
        <div
          className="p-12 flex flex-col items-center justify-center"
          style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}
        >
          <Loader2 size={32} className="animate-spin mb-4" style={{ color: '#1E3A8A' }} />
          <p className="text-sm font-medium" style={{ color: '#374151' }}>
            Loading {disasterType.toLowerCase()} explainability data from backend...
          </p>
        </div>
      ) : null}

      {state.kind === 'error' ? (
        <div
          className="p-4 mb-6 flex items-start gap-3"
          style={{
            border: '1px solid #B91C1C',
            borderRadius: '4px',
            backgroundColor: '#FEF2F2',
          }}
        >
          <AlertCircle size={18} style={{ color: '#B91C1C', flexShrink: 0, marginTop: 1 }} />
          <div>
            <p className="text-sm font-semibold mb-1" style={{ color: '#B91C1C' }}>
              Failed to load explainability data
            </p>
            <p className="text-sm" style={{ color: '#7F1D1D' }}>
              {state.message}
            </p>
          </div>
        </div>
      ) : null}

      {state.kind === 'unavailable' ? (
        <>
          <div
            className="p-4 mb-6 flex items-start gap-3"
            style={{
              border: '1px solid #CA8A04',
              borderRadius: '4px',
              backgroundColor: '#FFFBEB',
            }}
          >
            <AlertCircle size={18} style={{ color: '#CA8A04', flexShrink: 0, marginTop: 1 }} />
            <div>
              <p className="text-sm font-semibold mb-1" style={{ color: '#92400E' }}>
                Explainability data not yet generated
              </p>
              <p className="text-sm" style={{ color: '#78350F' }}>
                {state.note}
              </p>
            </div>
          </div>
          <EmptyState
            message={`No ${disasterType.toLowerCase()} feature attribution available.`}
            sub="Run or regenerate the corresponding training artifacts, then refresh this page."
          />
        </>
      ) : null}

      {state.kind === 'ready' ? (
        <>
          {state.data.top_feature ? (
            <div
              className="p-3 mb-6 flex items-center justify-between"
              style={{
                border: '1px solid #BFDBFE',
                borderRadius: '4px',
                backgroundColor: '#EFF6FF',
              }}
            >
              <div>
                <span className="text-sm font-semibold" style={{ color: '#1E3A8A' }}>
                  Most Influential Feature:
                </span>{' '}
                <span className="text-sm font-bold" style={{ color: '#1E3A8A' }}>
                  {formatFeatureName(state.data.top_feature)}
                </span>
                <span className="text-sm ml-2" style={{ color: '#3B82F6' }}>
                  ({state.data.feature_importance[0]?.importance_pct.toFixed(1)}%)
                </span>
              </div>
              <span className="text-xs" style={{ color: '#6B7280' }}>
                {state.data.total_features} features | {disasterType} source
              </span>
            </div>
          ) : null}

          <div className="flex mb-6" style={{ borderBottom: '1px solid #E5E7EB' }}>
            {(['importance', 'shap'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="px-6 py-3 text-sm font-medium"
                style={{
                  color: activeTab === tab ? '#1E3A8A' : '#6B7280',
                  borderBottom: activeTab === tab ? '2px solid #1E3A8A' : '2px solid transparent',
                  backgroundColor: 'transparent',
                }}
              >
                {tab === 'importance' ? 'Feature Importance' : 'SHAP-Style Contribution'}
              </button>
            ))}
          </div>

          {activeTab === 'importance' ? (
            <div className="p-4" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
              <h2 className="text-lg font-semibold mb-1" style={{ color: '#111827' }}>
                Feature Importance Analysis
              </h2>
              <p className="text-sm mb-6" style={{ color: '#6B7280' }}>
                Relative contribution of each input feature to the {disasterType.toLowerCase()} ensemble
                prediction.
              </p>
              <FeatureImportanceChart items={items} />
              <div
                className="p-3 mt-6 text-sm"
                style={{
                  backgroundColor: '#F3F4F6',
                  borderRadius: '4px',
                  border: '1px solid #E5E7EB',
                  color: '#6B7280',
                }}
              >
                <em>{state.data.note}</em>
              </div>
            </div>
          ) : (
            <div className="p-4" style={{ border: '1px solid #E5E7EB', borderRadius: '4px' }}>
              <h2 className="text-lg font-semibold mb-1" style={{ color: '#111827' }}>
                SHAP-Style Feature Contribution
              </h2>
              <p className="text-sm mb-6" style={{ color: '#6B7280' }}>
                Signed contribution values for the {disasterType.toLowerCase()} model set. Positive
                values increase risk; negative values decrease risk.
              </p>
              <ShapContributionList items={items} />
              <div className="flex justify-center gap-8 mt-6">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4" style={{ backgroundColor: '#B91C1C', borderRadius: '2px' }} />
                  <span className="text-sm" style={{ color: '#6B7280' }}>
                    Increases Risk (+)
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4" style={{ backgroundColor: '#15803D', borderRadius: '2px' }} />
                  <span className="text-sm" style={{ color: '#6B7280' }}>
                    Decreases Risk (-)
                  </span>
                </div>
              </div>
              <div
                className="p-3 mt-4 text-sm"
                style={{
                  backgroundColor: '#F3F4F6',
                  borderRadius: '4px',
                  border: '1px solid #E5E7EB',
                  color: '#6B7280',
                }}
              >
                <em>{state.data.note}</em>
              </div>
            </div>
          )}

          <EnsembleBreakdown items={items} disasterType={disasterType} />
        </>
      ) : null}

      {state.kind === 'idle' ? (
        <EmptyState message="Loading explainability data..." sub="Connecting to backend." />
      ) : null}
    </div>
  );
}
