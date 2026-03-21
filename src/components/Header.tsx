import { useEffect, useState } from 'react';
import { Menu } from 'lucide-react';
import { fetchHealth } from '../services/api';

type ApiStatus = 'unknown' | 'online' | 'degraded' | 'offline';

interface HeaderProps {
  onMenuClick: () => void;
}

export function Header({ onMenuClick }: HeaderProps) {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('unknown');

  useEffect(() => {
    let mounted = true;

    async function checkHealth() {
      try {
        const res = await fetchHealth();
        if (!mounted) return;
        setApiStatus(res.status === 'healthy' ? 'online' : 'degraded');
      } catch {
        if (!mounted) return;
        setApiStatus('offline');
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 30_000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const statusIndicator = (() => {
    switch (apiStatus) {
      case 'online':
        return (
          <>
            <span aria-hidden="true">🟢</span>
            <span className="hidden sm:inline">Running</span>
          </>
        );
      case 'degraded':
        return (
          <>
            <span aria-hidden="true">🟡</span>
            <span className="hidden sm:inline">Degraded</span>
          </>
        );
      case 'offline':
        return (
          <>
            <span aria-hidden="true">🔴</span>
            <span className="hidden sm:inline">Offline</span>
          </>
        );
      default:
        return (
          <>
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: '#9CA3AF',
                verticalAlign: 'middle',
              }}
            />
            <span className="hidden sm:inline" style={{ color: '#D1D5DB' }}>
              Checking...
            </span>
          </>
        );
    }
  })();

  return (
    <header
      className="h-16 flex items-center justify-between px-4 lg:px-6"
      style={{ backgroundColor: '#1E3A8A' }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 text-white"
          aria-label="Open navigation menu"
        >
          <Menu size={24} />
        </button>
        <h1 className="text-white text-sm lg:text-lg font-semibold">
          Ensemble-Based Hybrid Disaster Prediction System
        </h1>
      </div>

      <div className="flex items-center gap-4 lg:gap-6 text-white text-xs lg:text-sm">
        <div className="flex items-center gap-2">
          <span className="hidden sm:inline">API Status:</span>
          <span className="flex items-center gap-1">{statusIndicator}</span>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <span>Model Version:</span>
          <span className="font-mono">v1.0.0</span>
        </div>
      </div>
    </header>
  );
}
