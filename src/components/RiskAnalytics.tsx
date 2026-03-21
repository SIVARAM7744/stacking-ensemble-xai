import { BarChart2 } from 'lucide-react';

export function RiskAnalytics() {
  return (
    <section
      className="p-12 flex flex-col items-center justify-center text-center"
      style={{ border: '1px solid #E5E7EB', borderRadius: '4px', backgroundColor: '#FFFFFF' }}
    >
      <BarChart2 size={48} style={{ color: '#D1D5DB' }} className="mb-4" />
      <p className="text-lg font-semibold mb-1" style={{ color: '#374151' }}>
        Analytics unavailable — no prediction data yet.
      </p>
      <p className="text-sm" style={{ color: '#6B7280' }}>
        Charts and analytics will populate once predictions are made via the backend.
      </p>
    </section>
  );
}
