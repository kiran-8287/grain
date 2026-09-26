import React from 'react';
import { ShieldCheck, RefreshCw, Layers } from 'lucide-react';

interface NavbarProps {
  onReset: () => void;
  hasResult: boolean;
  isBackendHealthy: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ onReset, hasResult, isBackendHealthy }) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-600 to-amber-400 flex items-center justify-center shadow-lg shadow-amber-500/20">
            <span className="text-xl font-bold text-slate-950">🌾</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">Rice Quality AI</span>
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                KMS 2026-27
              </span>
            </div>
            <p className="text-xs text-slate-400">Raw Milled Rice 14-Parameter Quality Assessment</p>
          </div>
        </div>

        {/* Status & Actions */}
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60">
            <span className={`w-2 h-2 rounded-full ${isBackendHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
            <span>Backend: {isBackendHealthy ? 'Online' : 'Disconnected'}</span>
          </div>

          <div className="hidden md:flex items-center gap-1.5 text-xs text-slate-400 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60">
            <Layers className="w-3.5 h-3.5 text-amber-400" />
            <span>Multi-Label Defect Engine</span>
          </div>

          {hasResult && (
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-200 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors shadow-sm"
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
