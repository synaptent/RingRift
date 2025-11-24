import React from 'react';
import { reportClientError } from '../utils/errorReporting';

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

/**
 * Top-level React error boundary for catching render errors in the SPA.
 *
 * - Logs errors via the centralized reportClientError helper.
 * - Renders a minimal, user-friendly fallback UI.
 * - Intentionally avoids any complex logic to keep failure modes simple.
 */
class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    void reportClientError(error, {
      type: 'react_render_error',
      componentStack: errorInfo.componentStack,
    });
  }

  private handleReload = () => {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-slate-950 text-slate-100 px-4">
          <h1 className="text-2xl font-semibold mb-2">Something went wrong</h1>
          <p className="text-slate-300 mb-4 text-center max-w-md">
            An unexpected error occurred while rendering this page. Try refreshing the page.
          </p>
          <button
            type="button"
            onClick={this.handleReload}
            className="rounded bg-sky-600 px-4 py-2 text-sm font-medium hover:bg-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
          >
            Reload page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
