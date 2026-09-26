import React from 'react';
import { Download, FileSpreadsheet, FileCode, RotateCcw } from 'lucide-react';

interface ExportControlsProps {
  jobId?: string;
  onReset: () => void;
}

export const ExportControls: React.FC<ExportControlsProps> = ({ jobId, onReset }) => {
  const downloadFile = (endpoint: string, filename: string) => {
    if (!jobId) return;
    const url = `/api/analysis/${jobId}/export/${endpoint}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-wrap items-center justify-between gap-4">
      <div>
        <h4 className="text-sm font-bold text-white">Export Quality Report</h4>
        <p className="text-xs text-slate-400">Download complete grain measurements, defect labels, and sample statistics.</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => downloadFile('csv', `rice_analysis_${jobId?.slice(0, 8) ?? 'report'}.csv`)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition shadow-sm"
        >
          <FileSpreadsheet className="w-4 h-4" />
          <span>Download CSV</span>
        </button>

        <button
          onClick={() => downloadFile('json', `rice_analysis_${jobId?.slice(0, 8) ?? 'report'}.json`)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shadow-sm"
        >
          <FileCode className="w-4 h-4" />
          <span>Download JSON</span>
        </button>

        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition"
        >
          <RotateCcw className="w-4 h-4" />
          <span>New Analysis</span>
        </button>
      </div>
    </div>
  );
};
