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
  'Computing image fractions & geometry diagnostics',
  'Comparing with historical/reference limits',
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
      <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>

      <h2 className="text-xl font-bold text-text-primary mb-2">Analyzing Rice Sample</h2>
      <p className="text-xs text-text-secondary mb-8 max-w-sm mx-auto">
        Running computer vision algorithms and machine learning models across your image...
      </p>

      {/* Progress Steps */}
      <div className="bg-surface border border-border rounded-2xl p-6 text-left shadow-sm">
        <div className="space-y-3.5">
          {STAGES.map((stage, idx) => {
            const isDone = idx < activeStep;
            const isCurrent = idx === activeStep;
            return (
              <div
                key={stage}
                className={`flex items-center gap-3 text-xs transition-colors duration-200 ${
                  isDone
                    ? 'text-success font-medium'
                    : isCurrent
                    ? 'text-warning font-semibold'
                    : 'text-text-muted'
                }`}
              >
                <div className="w-5 h-5 flex items-center justify-center shrink-0">
                  {isDone ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 animate-spin text-warning" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-border-dark" />
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
