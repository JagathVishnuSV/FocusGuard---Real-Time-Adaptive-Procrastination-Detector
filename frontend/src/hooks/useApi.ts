import useSWR, { useSWRConfig } from 'swr';
import React from 'react';
import api from '../lib/api';
import type { SessionStatus } from '../lib/types';

// Generic fetcher for useSWR
const fetcher = (url: string) => api.get(url).then(res => res.data);

// Hook for most API GET requests
export function useApi<T>(endpoint: string | null) {
  const { data, error, isLoading } = useSWR<T>(endpoint, fetcher, {
    refreshInterval: 5000, // Refresh every 5 seconds
  });
  return { data, error, isLoading };
}

// Specialized hook for session management
export function useSessionStatus() {
  const { data, error, isLoading, mutate } = useSWR<SessionStatus>('/api/session/status', fetcher, {
    revalidateOnFocus: true,
    refreshInterval: 2500, // Check session status more frequently
  });
  const { mutate: globalMutate } = useSWRConfig();
  const [isMutating, setIsMutating] = React.useState(false);

  const startSession = async () => {
    setIsMutating(true);
    try {
      const response = await api.post('/api/session/start');
      const payload = response.data as { session?: SessionStatus } | undefined;
      mutate(payload?.session, { revalidate: false });
      globalMutate((key: unknown) => typeof key === 'string' && key.startsWith('/api/'), undefined, { revalidate: true });
    } catch (e) {
      console.error("Failed to start session", e);
      throw e;
    } finally {
      setIsMutating(false);
    }
  };

  const stopSession = async () => {
    setIsMutating(true);
    try {
      const response = await api.post('/api/session/stop');
      const payload = response.data as { session?: SessionStatus } | undefined;
      mutate(payload?.session, { revalidate: false });
      globalMutate((key: unknown) => typeof key === 'string' && key.startsWith('/api/'), undefined, { revalidate: true });
    } catch (e) {
      console.error("Failed to stop session", e);
      throw e;
    } finally {
      setIsMutating(false);
    }
  };

  return {
    sessionStatus: data,
    isLoading,
    isMutating,
    error,
    startSession,
    stopSession,
  };
}
