import { useState } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';

import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { PredictDisaster } from './pages/PredictDisaster';
import { Explainability } from './pages/Explainability';
import { HistoricalRecords } from './pages/HistoricalRecords';
import { APIStatus } from './pages/APIStatus';

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-white" style={{ color: '#111827' }}>
        {/* ── Top Header ───────────────────────────────────────────── */}
        <Header onMenuClick={() => setSidebarOpen((o) => !o)} />

        <div className="flex flex-1">
          {/* ── Left Sidebar ─────────────────────────────────────────── */}
          <Sidebar
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
          />

          {/* ── Main Content ─────────────────────────────────────────── */}
          <main className="flex-1 p-6 lg:ml-64 pb-24">
            <Routes>
              <Route path="/"               element={<Dashboard />} />
              <Route path="/predict"        element={<PredictDisaster />} />
              <Route path="/explainability" element={<Explainability />} />
              <Route path="/history"        element={<HistoricalRecords />} />
              <Route path="/api-status"     element={<APIStatus />} />
              {/* Catch-all → Dashboard */}
              <Route path="*"              element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>

        {/* ── Footer ───────────────────────────────────────────────── */}
        <footer
          className="fixed bottom-0 left-0 right-0 lg:left-64 p-4 text-center"
          style={{
            backgroundColor: '#F3F4F6',
            borderTop: '1px solid #E5E7EB',
          }}
        >
          <p className="text-xs" style={{ color: '#6B7280' }}>
            This system integrates heterogeneous ensemble learning, stacking-based meta
            optimization, explainable AI mechanisms, structured Disaster Risk Index
            computation, and scalable FastAPI deployment for real-time inference.
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}
