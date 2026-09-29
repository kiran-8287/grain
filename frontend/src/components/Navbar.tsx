import React from 'react';
import { ShieldCheck, RefreshCw } from 'lucide-react';

interface NavbarProps {
  onReset: () => void;
  hasResult: boolean;
  isBackendHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ onReset, hasResult, isBackendHealthy }) => {
  return (
    <header className="border-b border-border bg-surface sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center">
          <img src="/logo.png" alt="Grain Quality Analyzer" className="h-16 w-auto object-contain" />
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 text-xs text-text-secondary bg-surface-subtle px-3 py-1.5 rounded-lg border border-border">
            <span className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-success' : 'bg-error'}`} />
            <span>Backend: {isBackendHealthy ? 'Online' : 'Disconnected'}</span>
          </div>

          {hasResult && (
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 text-xs font-medium text-text-secondary hover:text-text-primary bg-surface-subtle hover:bg-brand-light px-3 py-1.5 rounded-lg border border-border transition-colors"
              title="Analyze another image"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>New Analysis</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
