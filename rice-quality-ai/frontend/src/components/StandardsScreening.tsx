import React from 'react';
import { StandardsResult } from '../types';
import { ShieldCheck, AlertCircle, HelpCircle, FileText, CheckCircle2, XCircle } from 'lucide-react';

interface StandardsScreeningProps {
  standards: StandardsResult;
}

export const StandardsScreening: React.FC<StandardsScreeningProps> = ({ standards }) => {
  const { screening, official_grade } = standards;

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'WITHIN REFERENCE LIMIT':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle2 className="w-3 h-3" />
            Within Limit
          </span>
        );
      case 'EXCEEDS REFERENCE LIMIT':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <XCircle className="w-3 h-3" />
            Exceeds Limit
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-800 text-slate-400 border border-slate-700">
            <HelpCircle className="w-3 h-3" />
            Not Assessable
          </span>
        );
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">Official Standards Screening</span>
            <span className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded-full">
              Government of India KMS 2026-27
            </span>
          </div>
          <h3 className="text-lg font-bold text-white mt-1">Image-Based Standard Screening</h3>
          <p className="text-xs text-slate-400">
            Parameter-by-parameter screening against the Uniform Specification for Raw Rice (Kharif Marketing Season 2026-27).
          </p>
        </div>
      </div>

      {/* Screening Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
              <th className="py-2.5 px-3">Quality Parameter</th>
              <th className="py-2.5 px-3">Observed Value</th>
              <th className="py-2.5 px-3">KMS 2026-27 Limit</th>
              <th className="py-2.5 px-3">Status</th>
              <th className="py-2.5 px-3">Measurement Basis</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {Object.entries(screening).map(([key, item]) => {
              const formattedName = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
              return (
                <tr key={key} className="hover:bg-slate-800/30 transition">
                  <td className="py-3 px-3 font-medium text-slate-200">{formattedName}</td>
                  <td className="py-3 px-3 font-semibold text-white">{item.observed_value}</td>
                  <td className="py-3 px-3 text-slate-300 font-mono">{item.reference_limit}</td>
                  <td className="py-3 px-3">{getStatusBadge(item.status)}</td>
                  <td className="py-3 px-3 text-[11px] text-slate-400">{item.basis}</td>
                </tr>
              );
            })}

            {/* Moisture Row (Explicit Non-Measurement, Section 26 of tasks.txt) */}
            <tr className="hover:bg-slate-800/30 transition bg-slate-800/20">
              <td className="py-3 px-3 font-medium text-slate-300">Moisture Content</td>
              <td className="py-3 px-3 text-slate-400 italic">Not measurable from this image</td>
              <td className="py-3 px-3 text-slate-400 font-mono">14.0% max</td>
              <td className="py-3 px-3">{getStatusBadge('NOT ASSESSABLE')}</td>
              <td className="py-3 px-3 text-[11px] text-slate-400">
                Moisture requires an appropriate physical/laboratory measurement and is outside the current image-only pipeline.
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Section 24: Formal Official Grade Status (Honest about limitations) */}
      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-amber-400" />
          <h4 className="text-sm font-bold text-white">Formal Official Grade Status</h4>
        </div>
        <p className="text-xs text-amber-200/90 font-medium">
          {official_grade.status}
        </p>
        <p className="text-xs text-slate-400 leading-relaxed">
          {official_grade.reason}
        </p>
        {official_grade.disclaimer && (
          <p className="text-[11px] text-slate-400 border-t border-slate-800/80 pt-2 italic">
            Note: {official_grade.disclaimer}
          </p>
        )}
      </div>
    </div>
  );
};
