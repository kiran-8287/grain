import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle, Clock } from 'lucide-react';

interface ProcessingStateProps {
  currentStage?: string;
}

const STAGES = [
  'Validating image & EXIF integrity',
  'Detecting rice presence & object bounds',
  'Segmenting individual grains & contours',
  'Measuring per-grain geometry (Length, Breadth, L/B)',
  'Classifying multi-label grain defects',
  'Detecting full-image foreign matter objects',
  'Computing sample-level statistics & admixture',
  'Comparing with India KMS 2026-27 standards',
  'Preparing annotated visualization & export report',
];

export const ProcessingState: React.FC<ProcessingStateProps> = () => {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev));
    }, 450);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-xl mx-auto px-4 py-16 text-center">
      <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>

      <h2 className="text-xl font-bold text-white mb-2">Analyzing Rice Sample</h2>
      <p className="text-xs text-slate-400 mb-8 max-w-sm mx-auto">
        Running computer vision algorithms and machine learning models across your image...
      </p>

      {/* Progress Steps */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 text-left shadow-2xl">
        <div className="space-y-3.5">
          {STAGES.map((stage, idx) => {
            const isDone = idx < activeStep;
            const isCurrent = idx === activeStep;
            return (
              <div
                key={stage}
                className={`flex items-center gap-3 text-xs transition-colors duration-200 ${
                  isDone
                    ? 'text-emerald-400 font-medium'
                    : isCurrent
                    ? 'text-amber-400 font-semibold'
                    : 'text-slate-500'
                }`}
              >
                <div className="w-5 h-5 flex items-center justify-center shrink-0">
                  {isDone ? (
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-slate-600" />
                  )}
                </div>
                <span>{stage}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
