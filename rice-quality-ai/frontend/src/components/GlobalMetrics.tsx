import React from 'react';
import { SampleSummary, ImageQuality } from '../types';
import { Layers, AlertTriangle, ShieldCheck, Activity, Award } from 'lucide-react';

interface GlobalMetricsProps {
  summary: SampleSummary;
  quality?: ImageQuality;
  warnings: string[];
}

export const GlobalMetrics: React.FC<GlobalMetricsProps> = ({ summary, quality, warnings }) => {
  const getQualityBadge = (tier?: string) => {
    switch (tier) {
      case 'GOOD':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">GOOD</span>;
      case 'FAIR':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">FAIR</span>;
      case 'POOR':
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">POOR</span>;
      default:
        return <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">UNRELIABLE</span>;
    }
  };

  return (
    <div className="space-y-4">
      {/* Small Sample Warning Banner (Sections 4 & 29 of tasks.txt) */}
      {summary.is_small_sample && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-200">
            <span className="font-semibold block text-amber-300">Statistical Small-Sample Notice:</span>
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
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">14. Total Count</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-white">{summary.total_rice_grains}</span>
            <span className="text-xs text-slate-400">grains</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">
            {summary.uncertain_grains > 0 ? `${summary.uncertain_grains} touching/merged` : 'Clean segmentation'}
          </span>
        </div>

        {/* Broken % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">1. Broken</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-amber-400">{summary.broken_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.broken_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Ref Limit: 25.0% max</span>
        </div>

        {/* Damaged % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">2. Damaged</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.damaged_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.damaged_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Ref Limit: 3.0% max</span>
        </div>

        {/* Discoloured % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">3. Discoloured</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.discoloured_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.discoloured_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Ref Limit: 3.0% max</span>
        </div>

        {/* Chalky % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">4. Chalky</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.chalky_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.chalky_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Ref Limit: 5.0% max</span>
        </div>

        {/* Red % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">5. Red Grain</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.red_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.red_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Ref Limit: 3.0% max</span>
        </div>

        {/* Dehusked % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">6. Dehusked</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.dehusked_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.dehusked_count})</span>
          </div>
          <span className="text-[10px] text-amber-400 mt-1 block">Visual proxy</span>
        </div>

        {/* Immature % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">7. Immature / Shrunken</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.immature_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.immature_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Geometry proxy</span>
        </div>

        {/* Sprouted / Weevilled % */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">8. Sprouted / Weevilled</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.sprouted_percent}%</span>
            <span className="text-xs text-slate-400 font-mono">({summary.sprouted_count})</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">ResNet-18 model</span>
        </div>

        {/* Foreign Matter */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">9. Foreign Matter</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.foreign_matter_count}</span>
            <span className="text-xs text-slate-400">objects</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Sample-level (Full image)</span>
        </div>

        {/* Admixture of Lower Class */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">10. Admixture</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-slate-100">{summary.admixture_percentage}%</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Sample-level (Mahalanobis)</span>
        </div>

        {/* Average Length */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">11. Length</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-slate-100">{summary.average_length}</span>
            <span className="text-xs text-slate-400 font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Median: {summary.median_length}</span>
        </div>

        {/* Average Breadth */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">12. Breadth</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-xl font-extrabold text-slate-100">{summary.average_breadth}</span>
            <span className="text-xs text-slate-400 font-mono">{summary.measurement_unit}</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Median: {summary.median_breadth}</span>
        </div>

        {/* Average L/B Ratio */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">13. L/B Ratio</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-amber-400">{summary.average_lb_ratio}</span>
          </div>
          <span className="text-[10px] text-slate-400 mt-1 block">Median: {summary.median_lb_ratio}</span>
        </div>

        {/* Image Quality Tier */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 shadow-sm flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Image Quality Tier</span>
          <div className="my-1">{getQualityBadge(quality?.tier)}</div>
          <span className="text-[10px] text-slate-400 block">Blur score: {Math.round(quality?.blur_score || 0)}</span>
        </div>
      </div>
    </div>
  );
};
