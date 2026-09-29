import React from 'react';
import { SampleSummary } from '../types';

interface GlobalMetricsProps {
  summary: SampleSummary;
  warnings: string[];
}

export const GlobalMetrics: React.FC<GlobalMetricsProps> = ({ summary, warnings }) => {
  const formatValue = (value: number | null | undefined, decimals = 2) =>
    value == null ? 'N/A' : `${Number(value).toFixed(decimals)}`;

  return (
    <div className="space-y-4">
      {/* Phase label */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-widest text-brand font-bold bg-brand-light border border-brand/20 px-2.5 py-1 rounded-full">
          Current Analysis
        </span>
        <span className="text-xs text-text-secondary font-medium">
          Rice Grain Segmentation &amp; Mask-based Geometry
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Total Count */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">Total Grains</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{summary.total_rice_grains}</span>
            <span className="text-xs text-text-muted">grains</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">
            {summary.uncertain_grains > 0 ? `${summary.uncertain_grains} touching/merged` : 'Clean segmentation'}
          </span>
        </div>

        {/* Average Length */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">Avg Length</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-text-primary">{formatValue(summary.average_length)}</span>
            <span className="text-xs text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {formatValue(summary.median_length)}</span>
        </div>

        {/* Average Breadth */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">Avg Breadth</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-text-primary">{formatValue(summary.average_breadth)}</span>
            <span className="text-xs text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {formatValue(summary.median_breadth)}</span>
        </div>

        {/* Average L/B Ratio */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">Avg L/B Ratio</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatValue(summary.average_lb_ratio)}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {formatValue(summary.median_lb_ratio)}</span>
        </div>
      </div>
    </div>
  );
};
