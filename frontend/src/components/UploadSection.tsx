import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, CheckCircle2, FileUp } from 'lucide-react';

interface UploadSectionProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
}

export const UploadSection: React.FC<UploadSectionProps> = ({ onAnalyze, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
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

    onAnalyze(selectedFile);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Intro Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-text-primary tracking-tight sm:text-4xl">
          Rice Grain Instance Segmentation
        </h1>
        <p className="text-sm text-text-secondary mt-2 max-w-2xl mx-auto">
          Upload an image to detect, segment, and count individual rice grains.
          Mask-based geometric measurements are extracted for each detected grain.
        </p>
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

        {/* Calibration Info */}
        <div className="text-xs text-text-secondary bg-surface p-2.5 rounded-lg border border-border flex items-start gap-2">
          <CheckCircle2 className="w-4 h-4 text-success shrink-0 mt-0.5" />
          <div>
            <span className="font-medium text-text-primary">Automatic Reference Detection Active:</span>
            <span className="block text-[11px] text-text-muted mt-0.5">
              ArUco markers are detected automatically. If absent, measurements are accurately reported in pixels without inventing fake millimeters.
            </span>
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
            <span>{isLoading ? 'Processing Analysis...' : 'Run Segmentation Analysis'}</span>
          </button>
        </div>
      </form>
    </div>
  );
};
