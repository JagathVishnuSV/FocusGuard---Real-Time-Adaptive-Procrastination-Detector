import React from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';

interface StatusDisplayProps {
  isLoading?: boolean;
  error?: unknown;
  data?: any[] | Record<string, unknown> | null;
  emptyMessage?: string;
  errorMessage?: string;
  className?: string;
  children?: React.ReactNode;
}

export function StatusDisplay({
  isLoading = false,
  error,
  data,
  emptyMessage = "No data available.",
  errorMessage = "Failed to load data.",
  className = "",
  children,
}: StatusDisplayProps) {
  if (isLoading) {
    return (
      <div className={`flex items-center justify-center p-8 text-muted-foreground ${className}`}>
        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
        <span>Loading...</span>
      </div>
    );
  }

  if (error) {
    const detail = error instanceof Error ? error.message : String(error ?? 'Unknown error');

    return (
      <div className={`flex flex-col items-center justify-center p-8 text-destructive ${className}`}>
        <AlertCircle className="mr-2 h-6 w-6" />
        <span className="font-semibold">{errorMessage}</span>
        <p className="text-xs text-muted-foreground mt-1">{detail || 'Please check the backend server.'}</p>
      </div>
    );
  }

  const isEmpty = Array.isArray(data) ? data.length === 0 : !data;

  if (isEmpty) {
    return (
      <div className={`flex items-center justify-center p-8 text-muted-foreground ${className}`}>
        <p>{emptyMessage}</p>
      </div>
    );
  }

  return <>{children}</>;
}
