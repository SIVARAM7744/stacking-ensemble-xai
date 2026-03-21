import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, BarChart2, Loader2 } from 'lucide-react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import { fetchHistory, type HistoryRecord } from '../services/api';

const pipelineSteps = [
  'Environmental Input',
  'Preprocessing',
  'RF / GB / SVM',
  'Stacking Meta-Learner',
  'Probability',
  'Disaster Risk Index',
  'Explainability',
];

const riskColors: Record<string, string> = {
  LOW: '#15803D',
  MODERATE: '#CA8A04',
  HIGH: '#B91C1C',
};

function getRiskBucket(level: string | null): 'LOW' | 'MODERATE' | 'HIGH' {
  const normalized = (level ?? '').toUpperCase();
  if (normalized === 'HIGH') return 'HIGH';
  if (normalized === 'MODERATE') return 'MODERATE';
  return 'LOW';
}

function toFloat(v: number | null | undefined): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

export function Dashboard() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchHistory({ page: 1, page_size: 200 });
        setRecords(res.records ?? []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const analytics = useMemo(() => {
    const total = records.length;
    const riskCount = { LOW: 0, MODERATE: 0, HIGH: 0 };
    const disasterCount = { Flood: 0, Earthquake: 0 };
    let sumDri = 0;
    let sumConfidence = 0;

    const trend = records
      .slice()
      .reverse()
      .map((r, i) => {
        const bucket = getRiskBucket(r.risk_level);
        riskCount[bucket] += 1;
        if (r.disaster_type === 'Flood') disasterCount.Flood += 1;
        if (r.disaster_type === 'Earthquake') disasterCount.Earthquake += 1;
        sumDri += toFloat(r.dri);
        sumConfidence += toFloat(r.confidence);
        return {
          index: i + 1,
          dri: Number(toFloat(r.dri).toFixed(4)),
          confidence: Number(toFloat(r.confidence).toFixed(1)),
          risk: bucket,
        };
      });

    return {
      total,
      latest: total > 0 ? records[0] : null,
      avgDri: total > 0 ? sumDri / total : 0,
      avgConfidence: total > 0 ? sumConfidence / total : 0,
      riskCount,
      disasterCount,
      trend: trend.length > 0 ? trend : [{ index: 1, dri: 0, confidence: 0, risk: 'LOW' }],
    };
  }, [records]);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold mb-2" style={{ color: '#111827' }}>
          Ensemble-Based Hybrid Disaster Prediction System
        </h1>
        <p className="text-sm" style={{ color: '#6B7280' }}>
          Live prediction analytics from stored model inference history
        </p>
      </div>

      {loading && (
        <section
          className="p-12 flex flex-col items-center justify-center text-center"
          style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
        >
          <Loader2 size={40} className="animate-spin mb-3" style={{ color: '#1E3A8A' }} />
          <p className="text-sm" style={{ color: '#6B7280' }}>Loading dashboard data...</p>
        </section>
      )}

      {!loading && error && (
        <section
          className="p-6 flex items-start gap-3"
          style={{ border: '1px solid #B91C1C', borderRadius: '4px', backgroundColor: '#FEF2F2' }}
        >
          <AlertCircle size={20} style={{ color: '#B91C1C' }} />
          <div>
            <p className="font-semibold" style={{ color: '#B91C1C' }}>Dashboard data unavailable</p>
            <p className="text-sm" style={{ color: '#7F1D1D' }}>{error}</p>
          </div>
        </section>
      )}

      {!loading && !error && (
        <>
          <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Total Predictions', value: String(analytics.total) },
              { label: 'Average DRI', value: analytics.avgDri.toFixed(4) },
              { label: 'Average Confidence', value: `${analytics.avgConfidence.toFixed(1)}%` },
              {
                label: 'Latest Risk',
                value: analytics.latest?.risk_level ?? 'N/A',
                color: riskColors[getRiskBucket(analytics.latest?.risk_level ?? null)],
              },
            ].map((item) => (
              <div
                key={item.label}
                className="p-4"
                style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
              >
                <div className="text-xs mb-1" style={{ color: '#6B7280' }}>{item.label}</div>
                <div className="text-xl font-semibold" style={{ color: item.color ?? '#111827' }}>{item.value}</div>
              </div>
            ))}
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}>
              <div className="px-4 py-3 border-b" style={{ borderColor: '#E5E7EB' }}>
                <h2 className="text-sm font-semibold" style={{ color: '#111827' }}>DRI Trend</h2>
              </div>
              <div className="h-64 p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={analytics.trend}>
                    <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                    <XAxis dataKey="index" tick={{ fill: '#6B7280', fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#6B7280', fontSize: 11 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="dri" stroke="#1E3A8A" strokeWidth={2} dot />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}>
              <div className="px-4 py-3 border-b" style={{ borderColor: '#E5E7EB' }}>
                <h2 className="text-sm font-semibold" style={{ color: '#111827' }}>Risk Distribution</h2>
              </div>
              <div className="h-64 p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'LOW', value: analytics.riskCount.LOW },
                        { name: 'MODERATE', value: analytics.riskCount.MODERATE },
                        { name: 'HIGH', value: analytics.riskCount.HIGH },
                      ]}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      <Cell fill={riskColors.LOW} />
                      <Cell fill={riskColors.MODERATE} />
                      <Cell fill={riskColors.HIGH} />
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <section style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}>
            <div className="px-4 py-3 border-b" style={{ borderColor: '#E5E7EB' }}>
              <h2 className="text-sm font-semibold" style={{ color: '#111827' }}>Disaster Type Distribution</h2>
            </div>
            <div className="h-64 p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    { name: 'Flood', count: analytics.disasterCount.Flood },
                    { name: 'Earthquake', count: analytics.disasterCount.Earthquake },
                  ]}
                >
                  <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: '#6B7280', fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fill: '#6B7280', fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1E3A8A" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </>
      )}

      <section
        className="p-4"
        style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
      >
        <h2 className="text-lg font-semibold mb-4" style={{ color: '#111827' }}>
          System Pipeline Architecture
        </h2>
        <div className="flex flex-col items-center py-4">
          {pipelineSteps.map((step, index) => (
            <div key={step} className="flex flex-col items-center">
              <div
                className="px-6 py-3 text-center min-w-48"
                style={{
                  border: '1px solid #1E3A8A',
                  borderRadius: '4px',
                  backgroundColor: index === 3 ? '#1E3A8A' : '#FFFFFF',
                  color: index === 3 ? '#FFFFFF' : '#1E3A8A',
                }}
              >
                <span className="text-sm font-medium">{step}</span>
              </div>
              {index < pipelineSteps.length - 1 && <div className="h-6 w-px" style={{ backgroundColor: '#1E3A8A' }} />}
              {index < pipelineSteps.length - 1 && (
                <div
                  className="w-0 h-0"
                  style={{
                    borderLeft: '6px solid transparent',
                    borderRight: '6px solid transparent',
                    borderTop: '8px solid #1E3A8A',
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </section>

      {!loading && !error && analytics.total === 0 && (
        <section
          className="p-8 flex flex-col items-center justify-center text-center"
          style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
        >
          <BarChart2 size={36} style={{ color: '#D1D5DB' }} className="mb-3" />
          <p className="text-sm font-semibold" style={{ color: '#374151' }}>
            Waiting for first prediction.
          </p>
          <p className="text-xs" style={{ color: '#6B7280' }}>
            Dashboard modules are active and will update immediately after one saved prediction.
          </p>
        </section>
      )}
    </div>
  );
}
