import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Sliders, CheckCircle2, AlertCircle, FileUp } from 'lucide-react';

interface UploadSectionProps {
  onAnalyze: (file: File, options: { grade: string; referencePixels?: number; referenceMm?: number }) => void;
  isLoading: boolean;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ onAnalyze, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [grade, setGrade] = useState<'grade_a' | 'common'>('grade_a');
  const [enableManualScale, setEnableManualScale] = useState(false);
  const [refPixels, setRefPixels] = useState<string>('100');
  const [refMm, setRefMm] = useState<string>('10');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setSelectedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    onAnalyze(selectedFile, {
      grade,
      referencePixels: enableManualScale ? parseFloat(refPixels) : undefined,
      referenceMm: enableManualScale ? parseFloat(refMm) : undefined,
    });
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Intro Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-text-primary tracking-tight sm:text-4xl">
          Automated Rice Grain Quality Analysis
        </h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? 'border-brand bg-brand-light'
              : 'border-border bg-surface hover:border-border-dark hover:bg-surface-subtle'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.tiff,.tif"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />

          {previewUrl ? (
            <div className="flex flex-col items-center gap-4">
              <div className="relative group max-w-sm rounded-xl overflow-hidden border border-border bg-surface-subtle">
                <img src={previewUrl} alt="Upload Preview" className="max-h-72 object-contain mx-auto" />
                <div className="absolute inset-0 bg-text-primary/0 group-hover:bg-text-primary/10 transition-opacity flex items-center justify-center">
                  <span className="text-xs text-text-primary bg-surface px-3 py-1.5 rounded-lg border border-border">
                    Click or drop to replace
                  </span>
                </div>
              </div>
              <div className="text-xs text-text-secondary">
                <span className="font-semibold text-text-primary">{selectedFile?.name}</span> ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-2xl bg-brand-light border border-border flex items-center justify-center text-brand mb-1">
                <Upload className="w-8 h-8" />
              </div>
              <div className="text-sm font-medium text-text-primary">
                Drag and drop your rice image here, or <span className="text-brand underline underline-offset-2">browse</span>
              </div>
              <p className="text-xs text-text-muted max-w-md">
                Supports JPG, PNG, WEBP, and TIFF. No fixed megapixel gate; results still depend on grain pixels, focus, contrast, and segmentation.
              </p>
            </div>
          )}
        </div>

        {/* Options Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-surface-subtle border border-border rounded-xl p-4">
          {/* Target Standard Profile */}
          <div>
            <label className="block text-xs font-semibold text-text-primary uppercase tracking-wider mb-2">
              Standard Grade Reference
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setGrade('grade_a')}
                className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition ${
                  grade === 'grade_a'
                    ? 'border-brand bg-brand-light text-brand'
                    : 'border-border bg-surface text-text-secondary hover:text-text-primary'
                }`}
              >
                <div className="font-semibold">Grade A Rice</div>
                <div className="text-[10px] text-text-muted mt-0.5">Historical/reference limit (6.0%); current season unverified</div>
              </button>
              <button
                type="button"
                onClick={() => setGrade('common')}
                className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition ${
                  grade === 'common'
                    ? 'border-brand bg-brand-light text-brand'
                    : 'border-border bg-surface text-text-secondary hover:text-text-primary'
                }`}
              >
                <div className="font-semibold">Common Rice</div>
                <div className="text-[10px] text-text-muted mt-0.5">Historical/reference profile; current season unverified</div>
              </button>
            </div>
          </div>

          {/* Metric Calibration Toggle */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-text-primary uppercase tracking-wider">
                Metric Calibration
              </label>
              <button
                type="button"
                onClick={() => setEnableManualScale(!enableManualScale)}
                className="text-xs text-brand hover:underline flex items-center gap-1"
              >
                <Sliders className="w-3 h-3" />
                {enableManualScale ? 'Disable Manual Scale' : 'Set Manual Scale (mm)'}
              </button>
            </div>

            {enableManualScale ? (
              <div className="grid grid-cols-2 gap-2 bg-surface p-2 rounded-lg border border-border text-xs">
                <div>
                  <span className="text-[10px] text-text-muted block mb-1">Known Object Pixels</span>
                  <input
                    type="number"
                    value={refPixels}
                    onChange={(e) => setRefPixels(e.target.value)}
                    className="w-full bg-surface-subtle border border-border rounded px-2 py-1 text-text-primary focus:outline-none focus:border-brand"
                    placeholder="e.g. 100"
                  />
                </div>
                <div>
                  <span className="text-[10px] text-text-muted block mb-1">Physical Size (mm)</span>
                  <input
                    type="number"
                    value={refMm}
                    onChange={(e) => setRefMm(e.target.value)}
                    className="w-full bg-surface-subtle border border-border rounded px-2 py-1 text-text-primary focus:outline-none focus:border-brand"
                    placeholder="e.g. 10"
                  />
                </div>
              </div>
            ) : (
              <div className="text-xs text-text-secondary bg-surface p-2.5 rounded-lg border border-border flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-success shrink-0 mt-0.5" />
                <div>
                  <span className="font-medium text-text-primary">Automatic Reference Detection Active:</span>
                  <span className="block text-[11px] text-text-muted mt-0.5">
                    ArUco markers are detected automatically. If absent, measurements are accurately reported in pixels without inventing fake millimeters.
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-center">
          <button
            type="submit"
            disabled={!selectedFile || isLoading}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-200 ${
              !selectedFile || isLoading
                ? 'bg-surface-subtle text-text-muted cursor-not-allowed border border-border'
                : 'bg-brand hover:bg-brand-dark text-white shadow-sm'
            }`}
          >
            <FileUp className="w-4 h-4" />
            <span>{isLoading ? 'Processing Analysis...' : 'Run Quality Analysis'}</span>
          </button>
        </div>
      </form>
    </div>
  );
};
