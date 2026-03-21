/**
 * useApiHealth.ts
 * ---------------
 * Polls GET /health every 30 seconds and exposes the result to the UI.
 *
 * Returns
 * -------
 *   online       boolean | null   — null = initial check in progress
 *   apiVersion   string | null
 *   dbConnected  boolean | null   — whether MySQL is reachable (from backend)
 *   lastChecked  Date | null
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchHealth } from '../services/api';

const POLL_INTERVAL_MS = 30_000;

interface ApiHealthState {
  online: boolean | null;
  apiVersion: string | null;
  dbConnected: boolean | null;
  lastChecked: Date | null;
}

export function useApiHealth(): ApiHealthState {
  const [state, setState] = useState<ApiHealthState>({
    online: null,
    apiVersion: null,
    dbConnected: null,
    lastChecked: null,
  });

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const check = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setState({
        online: true,
        apiVersion: data.api_version ?? null,
        dbConnected: data.db_connected ?? null,
        lastChecked: new Date(),
      });
    } catch {
      setState((prev) => ({
        ...prev,
        online: false,
        dbConnected: null,
        lastChecked: new Date(),
      }));
    }
  }, []);

  useEffect(() => {
    check();

    timerRef.current = setInterval(check, POLL_INTERVAL_MS);

    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
    };
  }, [check]);

  return state;
}
