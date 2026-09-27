import React from 'react';
import { SampleSummary } from '../types';
import { Layers, ShieldCheck, Activity, Award } from 'lucide-react';

interface GlobalMetricsProps {
  summary: SampleSummary;
  warnings: string[];
}

export const GlobalMetrics: React.FC<GlobalMetricsProps> = ({ summary, warnings }) => {
  const formatPercent = (value: number | null | undefined) =>
    value == null ? 'N/A' : `${value}%`;

  return (
    <div className="space-y-4">
      {/* Main 14 Project Parameters Summary Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {/* Total Count */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">14. Total Count</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{summary.total_rice_grains}</span>
            <span className="text-xs text-text-muted">grains</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">
            {summary.uncertain_grains > 0 ? `${summary.uncertain_grains} touching/merged` : 'Clean segmentation'}
          </span>
        </div>

        {/* Broken % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">1. Broken</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-warning">{formatPercent(summary.broken_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.broken_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Geometry estimate; not weight-based</span>
        </div>

        {/* Damaged % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">2. Damaged</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.damaged_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.damaged_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Experimental model count; not weight-based</span>
        </div>

        {/* Discoloured % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">3. Discoloured</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.discoloured_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.discoloured_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Image count estimate; not weight-based</span>
        </div>

        {/* Chalky % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">4. Chalky</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.chalky_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.chalky_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">
            {summary.chalky_analyzed_count ?? summary.total_rice_grains} grains classified
          </span>
        </div>

        {/* Red % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">5. Red Grain</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.red_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.red_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Image count estimate; not weight-based</span>
        </div>

        {/* Dehusked % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">6. Dehusked</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.dehusked_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.dehusked_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Visual proxy</span>
        </div>

        {/* Immature % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">7. Immature / Shrunken</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.immature_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.immature_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Geometry proxy</span>
        </div>

        {/* Sprouted / Weevilled % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">8. Sprouted / Weevilled</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.sprouted_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.sprouted_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Experimental; model score uncalibrated</span>
        </div>

        {/* Foreign Matter */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">9. Foreign Matter</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{summary.foreign_matter_count}</span>
            <span className="text-xs text-text-muted">objects</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Sample-level (Full image)</span>
        </div>

        {/* Admixture of Lower Class */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">10. Admixture</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.admixture_percentage)}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Unsupported: outlier is not lower-class evidence</span>
          <span className="text-[10px] text-text-muted block">
            Geometry outliers: {summary.geometry_outlier_count ?? 'N/A'} (diagnostic only)
          </span>
        </div>

        {/* Average Length */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">11. Length</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-text-primary">{summary.average_length}</span>
            <span className="text-xs text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {summary.median_length}</span>
        </div>

        {/* Average Breadth */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">12. Breadth</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-text-primary">{summary.average_breadth}</span>
            <span className="text-xs text-text-muted font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {summary.median_breadth}</span>
        </div>

        {/* Average L/B Ratio */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">13. L/B Ratio</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-warning">{summary.average_lb_ratio}</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Median: {summary.median_lb_ratio}</span>
        </div>

      </div>
    </div>
  );
};
