/**
 * HistoricalRecords.tsx
 * ----------------------
 * Displays paginated prediction history fetched from GET /history.
 *
 * Data source: src/services/api.ts → fetchHistory()
 * No hardcoded records. No mock data. No demo entries.
 * Table renders only when the backend returns real records.
 */

import { useEffect, useState, useCallback } from 'react';
import {
  Database,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Download,
  AlertCircle,
} from 'lucide-react';
import {
  API_BASE_URL,
  fetchHistory,
  type HistoryRecord,
  type HistoryQueryParams,
} from '../services/api';

// ── Constants ─────────────────────────────────────────────────────────────

const PAGE_SIZE = 20;

type DisasterFilter = '' | 'Flood' | 'Earthquake';
type RiskFilter     = '' | 'LOW' | 'MODERATE' | 'HIGH';

// ── Helpers ───────────────────────────────────────────────────────────────

function riskBadgeStyle(level: string | null): React.CSSProperties {
  if (level === 'HIGH')     return { backgroundColor: '#FEE2E2', color: '#B91C1C', border: '1px solid #B91C1C' };
  if (level === 'MODERATE') return { backgroundColor: '#FEF9C3', color: '#CA8A04', border: '1px solid #CA8A04' };
  if (level === 'LOW')      return { backgroundColor: '#DCFCE7', color: '#15803D', border: '1px solid #15803D' };
  return { backgroundColor: '#F3F4F6', color: '#6B7280', border: '1px solid #D1D5DB' };
}

function formatTs(ts: string | null): string {
  if (!ts) return '—';
  try {
    return (
      new Date(ts).toLocaleString('en-IN', {
        year: 'numeric', month: 'short', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        timeZone: 'UTC',
      }) + ' UTC'
    );
  } catch {
    return ts;
  }
}

function fmt(v: number | null | undefined, decimals = 4): string {
  if (v === null || v === undefined) return '—';
  return v.toFixed(decimals);
}

// ── CSV Export ────────────────────────────────────────────────────────────

function exportToCSV(records: HistoryRecord[]): void {
  const headers = [
    'ID', 'Timestamp', 'Disaster Type', 'Predicted', 'Probability',
    'DRI', 'Risk Level', 'Confidence (%)', 'Location', 'Region Code',
    'RF Prob', 'GB Prob', 'SVM Prob', 'Stacking Prob',
  ];

  const rows = records.map(r => [
    r.id ?? '',
    r.timestamp ?? '',
    r.disaster_type,
    r.predicted_disaster ?? '',
    fmt(r.probability),
    fmt(r.dri),
    r.risk_level ?? '',
    r.confidence != null ? r.confidence.toFixed(1) : '—',
    r.location ?? '',
    r.region_code ?? '',
    fmt((r.model_probabilities as Record<string, number>)?.random_forest),
    fmt((r.model_probabilities as Record<string, number>)?.gradient_boosting),
    fmt((r.model_probabilities as Record<string, number>)?.svm),
    fmt((r.model_probabilities as Record<string, number>)?.stacking),
  ]);

  const csvContent = [headers, ...rows]
    .map(row =>
      row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(','),
    )
    .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `prediction_history_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Component ─────────────────────────────────────────────────────────────

export function HistoricalRecords() {
  // ── State ──────────────────────────────────────────────────────────────
  const [records,        setRecords]        = useState<HistoryRecord[]>([]);
  const [total,          setTotal]          = useState<number>(0);
  const [page,           setPage]           = useState<number>(1);
  const [loading,        setLoading]        = useState<boolean>(true);
  const [error,          setError]          = useState<string | null>(null);

  // filters
  const [disasterFilter, setDisasterFilter] = useState<DisasterFilter>('');
  const [riskFilter,     setRiskFilter]     = useState<RiskFilter>('');
  const [dateFrom,       setDateFrom]       = useState<string>('');
  const [dateTo,         setDateTo]         = useState<string>('');

  // ── Fetch via centralized api.ts service layer ─────────────────────────
  const loadHistory = useCallback(async (p: number = 1) => {
    setLoading(true);
    setError(null);

    const params: HistoryQueryParams = {
      page:      p,
      page_size: PAGE_SIZE,
    };
    if (disasterFilter) params.disaster_type = disasterFilter;
    if (riskFilter)     params.risk_level    = riskFilter;
    if (dateFrom)       params.date_from     = dateFrom;
    if (dateTo)         params.date_to       = dateTo;

    try {
      const data = await fetchHistory(params);
      setRecords(data.records);
      setTotal(data.total);
      setPage(data.page);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setRecords([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [disasterFilter, riskFilter, dateFrom, dateTo]);

  // Initial load + re-fetch on filter change
  useEffect(() => {
    loadHistory(1);
  }, [loadHistory]);

  // ── Pagination ─────────────────────────────────────────────────────────
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const handlePage = (p: number) => {
    if (p >= 1 && p <= totalPages) loadHistory(p);
  };

  // ── Shared styles ──────────────────────────────────────────────────────
  const labelStyle: React.CSSProperties = {
    fontSize: '11px', fontWeight: 600, letterSpacing: '0.05em',
    color: '#6B7280', textTransform: 'uppercase',
  };
  const selectStyle: React.CSSProperties = {
    border: '1px solid #D1D5DB', borderRadius: '4px',
    padding: '6px 10px', fontSize: '13px', color: '#111827',
    backgroundColor: '#FFFFFF', cursor: 'pointer',
  };
  const inputStyle: React.CSSProperties = {
    border: '1px solid #D1D5DB', borderRadius: '4px',
    padding: '6px 10px', fontSize: '13px', color: '#111827',
    backgroundColor: '#FFFFFF',
  };

  const hasFilters = !!(disasterFilter || riskFilter || dateFrom || dateTo);

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="max-w-full">

      {/* ── Page Header ── */}
      <div
        className="flex items-center justify-between mb-6"
        style={{ flexWrap: 'wrap', gap: '12px' }}
      >
        <div>
          <h1 className="text-2xl font-bold" style={{ color: '#111827' }}>
            Historical Prediction Records
          </h1>
          <p className="text-sm mt-1" style={{ color: '#6B7280' }}>
            Archive of all past disaster risk predictions stored in MySQL
          </p>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {/* Export CSV — only shown when real records are loaded */}
          {records.length > 0 && (
            <button
              onClick={() => exportToCSV(records)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                border: '1px solid #1E3A8A', borderRadius: '4px',
                padding: '7px 14px', fontSize: '13px', fontWeight: 500,
                color: '#1E3A8A', backgroundColor: '#FFFFFF', cursor: 'pointer',
              }}
            >
              <Download size={14} /> Export CSV
            </button>
          )}

          {/* Refresh */}
          <button
            onClick={() => loadHistory(page)}
            disabled={loading}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              border: '1px solid #D1D5DB', borderRadius: '4px',
              padding: '7px 14px', fontSize: '13px', fontWeight: 500,
              color: '#374151', backgroundColor: '#FFFFFF',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
            }}
          >
            <RefreshCw size={14} style={loading ? { animation: 'spin 0.8s linear infinite' } : {}} />
            Refresh
          </button>
        </div>
      </div>

      {/* ── Filters ── */}
      <section
        style={{
          border: '1px solid #E5E7EB', borderRadius: '4px',
          backgroundColor: '#F3F4F6', padding: '16px', marginBottom: '16px',
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'flex-end' }}>

          {/* Disaster Type */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={labelStyle}>Disaster Type</label>
            <select
              value={disasterFilter}
              onChange={e => setDisasterFilter(e.target.value as DisasterFilter)}
              style={selectStyle}
            >
              <option value="">All Types</option>
              <option value="Flood">Flood</option>
              <option value="Earthquake">Earthquake</option>
            </select>
          </div>

          {/* Risk Level */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={labelStyle}>Risk Level</label>
            <select
              value={riskFilter}
              onChange={e => setRiskFilter(e.target.value as RiskFilter)}
              style={selectStyle}
            >
              <option value="">All Levels</option>
              <option value="HIGH">HIGH</option>
              <option value="MODERATE">MODERATE</option>
              <option value="LOW">LOW</option>
            </select>
          </div>

          {/* Date From */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={labelStyle}>Date From</label>
            <input
              type="date"
              value={dateFrom}
              onChange={e => setDateFrom(e.target.value)}
              style={inputStyle}
            />
          </div>

          {/* Date To */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={labelStyle}>Date To</label>
            <input
              type="date"
              value={dateTo}
              onChange={e => setDateTo(e.target.value)}
              style={inputStyle}
            />
          </div>

          {/* Clear Filters */}
          {hasFilters && (
            <button
              onClick={() => {
                setDisasterFilter('');
                setRiskFilter('');
                setDateFrom('');
                setDateTo('');
              }}
              style={{
                border: '1px solid #D1D5DB', borderRadius: '4px',
                padding: '7px 14px', fontSize: '13px', color: '#6B7280',
                backgroundColor: '#FFFFFF', cursor: 'pointer',
              }}
            >
              Clear Filters
            </button>
          )}
        </div>
      </section>

      {/* ── Loading State ── */}
      {loading && (
        <div
          style={{
            border: '1px solid #E5E7EB', borderRadius: '4px',
            padding: '64px', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: '12px',
            backgroundColor: '#FFFFFF',
          }}
        >
          <div
            style={{
              width: '28px', height: '28px',
              border: '2px solid #E5E7EB',
              borderTop: '2px solid #1E3A8A',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
          <p style={{ color: '#6B7280', fontSize: '14px' }}>
            Loading prediction history…
          </p>
        </div>
      )}

      {/* ── Error State ── */}
      {!loading && error && (
        <div
          style={{
            border: '1px solid #B91C1C', borderRadius: '4px',
            padding: '24px', display: 'flex', alignItems: 'flex-start',
            gap: '12px', backgroundColor: '#FEF2F2', marginBottom: '16px',
          }}
        >
          <AlertCircle
            size={20}
            style={{ color: '#B91C1C', flexShrink: 0, marginTop: '2px' }}
          />
          <div>
            <p style={{ fontSize: '14px', fontWeight: 600, color: '#B91C1C', marginBottom: '4px' }}>
              Unable to load historical records
            </p>
            <p style={{ fontSize: '13px', color: '#7F1D1D' }}>{error}</p>
            <p style={{ fontSize: '12px', color: '#9CA3AF', marginTop: '6px' }}>
              Ensure the FastAPI backend is running at{' '}
              <code style={{ fontFamily: 'monospace' }}>{API_BASE_URL}</code>
            </p>
          </div>
        </div>
      )}

      {/* ── Empty State ── */}
      {!loading && !error && records.length === 0 && (
        <section
          style={{
            border: '1px solid #E5E7EB', borderRadius: '4px',
            padding: '64px', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            textAlign: 'center', backgroundColor: '#FFFFFF',
          }}
        >
          <Database size={48} style={{ color: '#D1D5DB', marginBottom: '16px' }} />
          <p style={{ fontSize: '16px', fontWeight: 600, color: '#374151', marginBottom: '6px' }}>
            No historical records available.
          </p>
          <p style={{ fontSize: '13px', color: '#6B7280' }}>
            {hasFilters
              ? 'No records match the selected filters. Try clearing the filters.'
              : 'Prediction history will appear here once predictions are made and saved to the database.'}
          </p>
        </section>
      )}

      {/* ── Results Table ── */}
      {!loading && !error && records.length > 0 && (
        <>
          {/* Summary bar */}
          <div
            style={{
              display: 'flex', justifyContent: 'space-between',
              alignItems: 'center', marginBottom: '8px',
            }}
          >
            <p style={{ fontSize: '13px', color: '#6B7280' }}>
              Showing{' '}
              <strong style={{ color: '#111827' }}>{records.length}</strong>{' '}
              of{' '}
              <strong style={{ color: '#111827' }}>{total}</strong>{' '}
              record{total !== 1 ? 's' : ''}
              {hasFilters && ' (filtered)'}
            </p>
            <p style={{ fontSize: '12px', color: '#9CA3AF' }}>
              Page {page} of {totalPages}
            </p>
          </div>

          {/* Scrollable table */}
          <div
            style={{
              overflowX: 'auto',
              border: '1px solid #E5E7EB',
              borderRadius: '4px',
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ backgroundColor: '#F3F4F6', borderBottom: '1px solid #E5E7EB' }}>
                  {[
                    'ID', 'Date', 'Disaster Type', 'Predicted',
                    'DRI', 'Probability', 'Risk Level', 'Confidence',
                    'Location', 'RF', 'GB', 'SVM', 'Stacking',
                  ].map(h => (
                    <th
                      key={h}
                      style={{
                        padding: '10px 12px', textAlign: 'left',
                        fontSize: '11px', fontWeight: 700,
                        color: '#6B7280', letterSpacing: '0.05em',
                        textTransform: 'uppercase', whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {records.map((r, idx) => {
                  const mp = r.model_probabilities as Record<string, number>;
                  return (
                    <tr
                      key={r.id ?? idx}
                      style={{
                        borderBottom: '1px solid #F3F4F6',
                        backgroundColor: idx % 2 === 0 ? '#FFFFFF' : '#FAFAFA',
                      }}
                    >
                      {/* ID */}
                      <td style={{ padding: '10px 12px', color: '#9CA3AF', fontFamily: 'monospace' }}>
                        #{r.id ?? '—'}
                      </td>

                      {/* Date */}
                      <td style={{ padding: '10px 12px', color: '#374151', whiteSpace: 'nowrap' }}>
                        {formatTs(r.timestamp)}
                      </td>

                      {/* Disaster Type */}
                      <td style={{ padding: '10px 12px', fontWeight: 500, color: '#1E3A8A' }}>
                        {r.disaster_type}
                      </td>

                      {/* Predicted */}
                      <td style={{ padding: '10px 12px', color: '#374151' }}>
                        {r.predicted_disaster ?? '—'}
                      </td>

                      {/* DRI */}
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#111827', fontWeight: 600 }}>
                        {fmt(r.dri)}
                      </td>

                      {/* Probability */}
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {fmt(r.probability)}
                      </td>

                      {/* Risk Level badge */}
                      <td style={{ padding: '10px 12px' }}>
                        <span
                          style={{
                            ...riskBadgeStyle(r.risk_level ?? null),
                            padding: '2px 8px', borderRadius: '4px',
                            fontSize: '11px', fontWeight: 700, whiteSpace: 'nowrap',
                          }}
                        >
                          {r.risk_level ?? '—'}
                        </span>
                      </td>

                      {/* Confidence */}
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {r.confidence != null ? `${r.confidence.toFixed(1)}%` : '—'}
                      </td>

                      {/* Location */}
                      <td style={{ padding: '10px 12px', color: '#6B7280' }}>
                        {r.location ?? '—'}
                      </td>

                      {/* Per-model probabilities */}
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {fmt(mp?.random_forest)}
                      </td>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {fmt(mp?.gradient_boosting)}
                      </td>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {fmt(mp?.svm)}
                      </td>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace', color: '#374151' }}>
                        {fmt(mp?.stacking)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ── Pagination ── */}
          {totalPages > 1 && (
            <div
              style={{
                display: 'flex', justifyContent: 'center',
                alignItems: 'center', gap: '8px', marginTop: '16px',
              }}
            >
              <button
                onClick={() => handlePage(page - 1)}
                disabled={page <= 1}
                style={{
                  border: '1px solid #D1D5DB', borderRadius: '4px',
                  padding: '6px 10px',
                  cursor: page <= 1 ? 'not-allowed' : 'pointer',
                  opacity: page <= 1 ? 0.4 : 1,
                  backgroundColor: '#FFFFFF',
                  display: 'flex', alignItems: 'center',
                }}
              >
                <ChevronLeft size={16} style={{ color: '#374151' }} />
              </button>

              {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
                let p: number;
                if (totalPages <= 7)          p = i + 1;
                else if (page <= 4)           p = i + 1;
                else if (page >= totalPages - 3) p = totalPages - 6 + i;
                else                          p = page - 3 + i;

                return (
                  <button
                    key={p}
                    onClick={() => handlePage(p)}
                    style={{
                      border: `1px solid ${p === page ? '#1E3A8A' : '#D1D5DB'}`,
                      borderRadius: '4px', padding: '6px 12px',
                      fontSize: '13px', fontWeight: p === page ? 700 : 400,
                      backgroundColor: p === page ? '#1E3A8A' : '#FFFFFF',
                      color: p === page ? '#FFFFFF' : '#374151',
                      cursor: 'pointer',
                    }}
                  >
                    {p}
                  </button>
                );
              })}

              <button
                onClick={() => handlePage(page + 1)}
                disabled={page >= totalPages}
                style={{
                  border: '1px solid #D1D5DB', borderRadius: '4px',
                  padding: '6px 10px',
                  cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                  opacity: page >= totalPages ? 0.4 : 1,
                  backgroundColor: '#FFFFFF',
                  display: 'flex', alignItems: 'center',
                }}
              >
                <ChevronRight size={16} style={{ color: '#374151' }} />
              </button>
            </div>
          )}
        </>
      )}

      {/* Spinner keyframe */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
