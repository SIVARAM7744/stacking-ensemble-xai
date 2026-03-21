/**
 * usePrediction
 * Encapsulates the ensemble prediction request lifecycle:
 * idle → loading → result | error
 */

import { useState, useCallback } from 'react';
import {
  API_BASE_URL,
  runPrediction,
  type PredictionRequest,
  type PredictionResponse,
} from '../services/api';

export type PredictionStatus = 'idle' | 'loading' | 'success' | 'error';

export interface UsePredictionResult {
  status: PredictionStatus;
  loadingStep: number;
  data: PredictionResponse | null;
  error: string | null;
  predict: (req: PredictionRequest) => Promise<void>;
  reset: () => void;
}

export function usePrediction(): UsePredictionResult {
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [loadingStep, setLoadingStep] = useState(0);
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (req: PredictionRequest) => {
    setStatus('loading');
    setData(null);
    setError(null);
    setLoadingStep(0);

    // Advance loading-step indicator every 400 ms for UI feedback
    let step = 0;
    const interval = setInterval(() => {
      step = Math.min(step + 1, 4);
      setLoadingStep(step);
    }, 400);

    try {
      const res = await runPrediction(req);
      clearInterval(interval);
      setLoadingStep(5);
      setData(res);
      setStatus('success');
    } catch (err) {
      clearInterval(interval);
      setError(
        err instanceof Error
          ? err.message
          : `Backend connection failed. Ensure API is running at ${API_BASE_URL}`
      );
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setLoadingStep(0);
    setData(null);
    setError(null);
  }, []);

  return { status, loadingStep, data, error, predict, reset };
}
