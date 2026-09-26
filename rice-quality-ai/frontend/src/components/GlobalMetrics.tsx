import React from 'react';
import { SampleSummary, ImageQuality } from '../types';
import { Layers, AlertTriangle, ShieldCheck, Activity, Award } from 'lucide-react';

interface GlobalMetricsProps {
  summary: SampleSummary;
  quality?: ImageQuality;
  warnings: string[];
}

export const GlobalMetrics: React.FC<GlobalMetricsProps> = ({ summary, quality, warnings }) => {
  const formatPercent = (value: number | null | undefined) =>
    value == null ? 'N/A' : `${value}%`;

  const getQualityBadge = (tier?: string) => {
    switch (tier) {
      case 'GOOD':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-success-bg text-success border border-success/20">GOOD</span>;
      case 'FAIR':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-info-bg text-info border border-info/20">FAIR</span>;
      case 'POOR':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-warning-bg text-warning border border-warning/20">POOR</span>;
      default:
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-error-bg text-error border border-error/20">UNRELIABLE</span>;
    }
  };

  return (
    <div className="space-y-4">
      {/* Small Sample Warning Banner */}
      {summary.is_small_sample && (
        <div className="bg-warning-bg border border-warning/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-warning shrink-0 mt-0.5" />
          <div className="text-xs text-warning">
            <span className="font-semibold block text-warning">Statistical Small-Sample Notice:</span>
            {summary.total_rice_grains === 1 ? (
              <p className="mt-0.5">
                Sample size is 1 grain. Individual-grain analysis is available, but sample-level quality percentages
                are not representative of a larger rice lot.
              </p>
            ) : (
              <p className="mt-0.5">
                Observed sample size ({summary.total_rice_grains} grains) is below statistical threshold (30 grains).
                Individual classifications are provided, but sample percentages reflect observed sample fraction, not whole batch quality.
              </p>
            )}
          </div>
        </div>
      )}

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
          <span className="text-[10px] text-text-muted mt-1 block">Image count estimate; not weight-based</span>
        </div>

        {/* Damaged % */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">2. Damaged</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-text-primary">{formatPercent(summary.damaged_percent)}</span>
            <span className="text-xs text-text-muted font-mono">({summary.damaged_count})</span>
          </div>
          <span className="text-[10px] text-text-muted mt-1 block">Image count estimate; not weight-based</span>
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
          <span className="text-[10px] text-text-muted mt-1 block">Sample-level (Mahalanobis)</span>
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

        {/* Image Quality Tier */}
        <div className="bg-surface border border-border rounded-xl p-3.5 shadow-sm flex flex-col justify-between">
          <span className="text-[11px] font-medium text-text-muted uppercase tracking-wider block">Image Quality Tier</span>
          <div className="my-1">{getQualityBadge(quality?.tier)}</div>
          <span className="text-[10px] text-text-muted block">Blur score: {Math.round(quality?.blur_score || 0)}</span>
        </div>
      </div>
    </div>
  );
};
