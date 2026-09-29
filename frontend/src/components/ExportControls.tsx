import React from 'react';
import { Download, FileSpreadsheet, FileCode, RotateCcw } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface ExportControlsProps {
  jobId?: string;
  onReset: () => void;
}

export const ExportControls: React.FC<ExportControlsProps> = ({ jobId, onReset }) => {
  const downloadFile = (endpoint: string, filename: string) => {
    if (!jobId) return;
    const url = `${API_BASE}/api/analysis/${jobId}/export/${endpoint}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm flex flex-wrap items-center justify-between gap-4">
      <div>
        <h4 className="text-sm font-bold text-text-primary">Export Current Analysis</h4>
        <p className="text-xs text-text-secondary">Download grain count, confidence, length, breadth, and L/B ratio.</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => downloadFile('csv', `rice_analysis_${jobId?.slice(0, 8) ?? 'report'}.csv`)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-brand hover:bg-brand-dark text-white transition shadow-sm"
        >
          <FileSpreadsheet className="w-4 h-4" />
          <span>Download CSV</span>
        </button>

        <button
          onClick={() => downloadFile('json', `rice_analysis_${jobId?.slice(0, 8) ?? 'report'}.json`)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-surface-subtle hover:bg-brand-light text-text-primary border border-border transition shadow-sm"
        >
          <FileCode className="w-4 h-4" />
          <span>Download JSON</span>
        </button>

        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-surface-subtle hover:bg-brand-light text-text-secondary hover:text-text-primary border border-border transition"
        >
          <RotateCcw className="w-4 h-4" />
          <span>New Analysis</span>
        </button>
      </div>
    </div>
  );
};
