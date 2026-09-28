import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { UploadSection } from './components/UploadSection';
import { ProcessingState } from './components/ProcessingState';
import { GlobalMetrics } from './components/GlobalMetrics';
import { AnnotatedViewer } from './components/AnnotatedViewer';
import { GrainDetailPanel } from './components/GrainDetailPanel';
import { StandardsScreening } from './components/StandardsScreening';
import { GrainTable } from './components/GrainTable';
import { QualityWarningsPanel } from './components/QualityWarningsPanel';
import { ExportControls } from './components/ExportControls';
import { AnalysisResult, GrainInstance } from './types';
import { AlertCircle, RefreshCw, XCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedGrainId, setSelectedGrainId] = useState<number | null>(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [nonRiceMessage, setNonRiceMessage] = useState<string | null>(null);
  const [nonRiceResult, setNonRiceResult] = useState<AnalysisResult | null>(null);

  const riceGate = nonRiceResult?.rice_gate;

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          setBackendHealthy(data.status === 'healthy');
        } else {
          setBackendHealthy(false);
        }
      } catch (err) {
        setBackendHealthy(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async (file: File) => {
    setIsLoading(true);
    setErrorMessage(null);
    setNonRiceMessage(null);
    setNonRiceResult(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('grade', 'grade_a');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data: AnalysisResult = await response.json();

      if (!data.rice_detected) {
        setNonRiceResult(data);
        setNonRiceMessage(data.message || 'No rice grains detected. Please upload an image containing rice grains.');
      } else {
        setResult(data);
        if (data.grains && data.grains.length > 0) {
          setSelectedGrainId(data.grains[0].id);
        }
      }
    } catch (err: any) {
      setErrorMessage(err.message || 'An unexpected error occurred during processing.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setSelectedGrainId(null);
    setErrorMessage(null);
    setNonRiceMessage(null);
    setNonRiceResult(null);
  };

  const selectedGrain: GrainInstance | null =
    result?.grains?.find((g) => g.id === selectedGrainId) || (result?.grains?.[0] ?? null);

  return (
    <div className="min-h-screen bg-surface-page text-text-primary flex flex-col">
      {/* Navbar */}
      <Navbar
        onReset={handleReset}
        hasResult={result !== null || nonRiceMessage !== null}
        isBackendHealthy={backendHealthy}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        {/* Error Banner */}
            {errorMessage && (
              <div className="mb-6 bg-error-bg border border-error/30 rounded-xl p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-error shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold text-error text-sm">Analysis Request Failed</span>
                  <p className="text-xs text-text-secondary mt-1">{errorMessage}</p>
                </div>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="text-xs text-error hover:text-text-primary underline"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Non-Rice Detected Message */}
            {nonRiceMessage && (
              <div className="max-w-xl mx-auto my-12 bg-notrice-bg border border-notrice-border rounded-2xl p-8 text-center space-y-4">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-notrice-bg border border-notrice-border flex items-center justify-center text-notrice-text">
                  <XCircle className="w-8 h-8" />
                </div>
                <div className="space-y-1">
                  <p className="text-[11px] uppercase tracking-widest text-text-muted font-semibold">
                    Rice Analysis
                  </p>
                  <h2 className="text-xl font-bold text-text-primary">NOT RICE</h2>
                </div>
                <p className="text-sm text-text-secondary leading-relaxed font-medium">
                  "{nonRiceMessage}"
                </p>
                {riceGate && (
                  <div className="grid grid-cols-3 gap-3 pt-1">
                    <div className="bg-surface border border-border rounded-xl px-3 py-3">
                      <p className="text-[10px] uppercase tracking-wide text-text-muted">Rice Grains</p>
                      <p className="text-lg font-bold text-text-primary">{riceGate.rice_detections}</p>
                    </div>
                    <div className="bg-surface border border-border rounded-xl px-3 py-3">
                      <p className="text-[10px] uppercase tracking-wide text-text-muted">Foreign Matter</p>
                      <p className="text-lg font-bold text-text-primary">{riceGate.foreign_matter_detections}</p>
                    </div>
                    <div className="bg-surface border border-border rounded-xl px-3 py-3">
                      <p className="text-[10px] uppercase tracking-wide text-text-muted">Objects Detected</p>
                      <p className="text-lg font-bold text-text-primary">{riceGate.total_detections}</p>
                    </div>
                  </div>
                )}
                {riceGate && (
                  <p className="text-xs text-warning font-semibold">
                    Rice analysis stopped — segmentation, per-grain measurements, defect
                    classification, grading and admixture were NOT executed.
                  </p>
                )}
                <p className="text-xs text-text-secondary max-w-md mx-auto">
                  The detector analysed every object in the image and compared it against the
                  rice class (
                  <span className="text-text-primary font-medium">{riceGate?.rice_class_name ?? 'rice_grain'}</span>
                  , confidence threshold{' '}
                  {riceGate?.rice_confidence_threshold ?? '—'}). Foreign matter is detected and
                  reported separately — it is never treated as rice. Fake measurements are
                  strictly prevented.
                </p>
                {riceGate?.model_data_provenance === 'synthetic_hand_sampled_feature_vectors_only' && (
                  <p className="text-xs text-warning max-w-md mx-auto">
                    Gate score status: Experimental / uncalibrated. The loaded feature model was trained on synthetic feature vectors, not a real labeled image corpus.
                  </p>
                )}
                {riceGate && (
                  <p className="text-[10px] font-mono text-text-muted break-words">{riceGate.debug}</p>
                )}
                <div className="pt-2">
                  <button
                    onClick={handleReset}
                    className="px-6 py-2.5 rounded-xl font-semibold text-xs bg-brand hover:bg-brand-dark text-white transition"
                  >
                    Upload Another Image
                  </button>
                </div>
              </div>
            )}

            {/* Processing State */}
            {isLoading && <ProcessingState />}

            {/* Upload View */}
            {!isLoading && !result && !nonRiceMessage && (
              <UploadSection onAnalyze={handleAnalyze} isLoading={isLoading} />
            )}

            {/* Results Dashboard */}
            {!isLoading && result && (
              <div className="space-y-6 animate-fadeIn">
                {/* 1. Global Metrics & 14 Project Parameters */}
                <GlobalMetrics
                  summary={result.summary}
                  warnings={result.warnings}
                />

                {/* 2. Full-width annotated image */}
                <AnnotatedViewer
                  annotatedImageUrl={result.annotated_image_base64}
                  originalImageUrl={result.original_image_base64}
                  grains={result.grains}
                  selectedGrainId={selectedGrainId}
                  onSelectGrain={(id) => setSelectedGrainId(id)}
                />

                {/* 3. Grain Population Table */}
                {result.grains && result.grains.length > 0 && (
                  <GrainTable
                    grains={result.grains}
                    selectedGrainId={selectedGrainId}
                    onSelectGrain={(id) => setSelectedGrainId(id)}
                    unit={result.summary.measurement_unit}
                  />
                )}

                {/* 4. Selected grain parameter summary */}
                <GrainDetailPanel
                  grain={selectedGrain}
                  unit={result.summary.measurement_unit}
                />

                {/* 5. Historical/reference image screening */}
                <StandardsScreening standards={result.standards} />

                {/* 6. Provenance & Diagnostics */}
                <QualityWarningsPanel
                  quality={result.quality}
                  warnings={result.warnings}
                  calibrationMode={result.calibration.mode}
                  pixelsPerMm={result.calibration.pixels_per_mm}
                  megapixels={result.image.megapixels}
                />

                {/* 7. Export Controls */}
                <ExportControls jobId={result.job_id} onReset={handleReset} />
              </div>
             )}
      </main>
    </div>
  );
};
