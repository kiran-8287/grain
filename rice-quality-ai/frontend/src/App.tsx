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
import { AlertCircle, RefreshCw, Layers } from 'lucide-react';

export const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedGrainId, setSelectedGrainId] = useState<number | null>(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [nonRiceMessage, setNonRiceMessage] = useState<string | null>(null);

  // Health check on mount
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

  const handleAnalyze = async (
    file: File,
    options: { grade: string; referencePixels?: number; referenceMm?: number }
  ) => {
    setIsLoading(true);
    setErrorMessage(null);
    setNonRiceMessage(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('grade', options.grade);
    if (options.referencePixels && options.referenceMm) {
      formData.append('reference_pixels', options.referencePixels.toString());
      formData.append('reference_mm', options.referenceMm.toString());
    }

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
        // Explicit prompt requirement:
        // "No rice grains detected. Please upload an image containing rice grains."
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
  };

  const selectedGrain: GrainInstance | null =
    result?.grains?.find((g) => g.id === selectedGrainId) || (result?.grains?.[0] ?? null);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
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
          <div className="mb-6 bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span className="font-semibold text-rose-300 text-sm">Analysis Request Failed</span>
              <p className="text-xs text-rose-200/90 mt-1">{errorMessage}</p>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-xs text-rose-300 hover:text-white underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Non-Rice Detected Message (Section 1 of tasks.txt) */}
        {nonRiceMessage && (
          <div className="max-w-xl mx-auto my-12 bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center space-y-4 shadow-2xl">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
              <AlertCircle className="w-8 h-8" />
            </div>
            <h2 className="text-xl font-bold text-white">No Rice Grains Detected</h2>
            <p className="text-sm text-slate-300 leading-relaxed font-medium">
              "{nonRiceMessage}"
            </p>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              The computer vision detector analyzed object aspect ratios, contours, and surface textures.
              No valid rice grains were found. Fake measurements are strictly prevented.
            </p>
            <div className="pt-2">
              <button
                onClick={handleReset}
                className="px-6 py-2.5 rounded-xl font-semibold text-xs bg-amber-500 hover:bg-amber-400 text-slate-950 transition shadow-lg shadow-amber-500/20"
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
              quality={result.quality}
              warnings={result.warnings}
            />

            {/* 2. Visual Inspection & Grain Detail Split */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left: Annotated Image Viewer (7 cols) */}
              <div className="lg:col-span-7">
                <AnnotatedViewer
                  annotatedImageUrl={result.annotated_image_base64}
                  grains={result.grains}
                  selectedGrainId={selectedGrainId}
                  onSelectGrain={(id) => setSelectedGrainId(id)}
                />
              </div>

              {/* Right: Selected Grain Detail Inspector (5 cols) */}
              <div className="lg:col-span-5">
                <GrainDetailPanel
                  grain={selectedGrain}
                  unit={result.summary.measurement_unit}
                />
              </div>
            </div>

            {/* 3. Grain Population Table */}
            {result.grains && result.grains.length > 0 && (
              <GrainTable
                grains={result.grains}
                selectedGrainId={selectedGrainId}
                onSelectGrain={(id) => setSelectedGrainId(id)}
                unit={result.summary.measurement_unit}
              />
            )}

            {/* 4. Official Standards Screening (India KMS 2026-27) */}
            <StandardsScreening standards={result.standards} />

            {/* 5. Provenance & Diagnostics */}
            <QualityWarningsPanel
              quality={result.quality}
              warnings={result.warnings}
              calibrationMode={result.calibration.mode}
              pixelsPerMm={result.calibration.pixels_per_mm}
              megapixels={result.image.megapixels}
            />

            {/* 6. Export Controls */}
            <ExportControls jobId={result.job_id} onReset={handleReset} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-400">
        Rice Quality AI &bull; Kharif Marketing Season 2026-27 Standards Screening &bull; Multi-Label Defect Model
      </footer>
    </div>
  );
};
