import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Sliders, CheckCircle2, AlertCircle, FileUp, Sparkles } from 'lucide-react';

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

  // Helper to generate a quick synthetic sample image directly in canvas for instant testing
  const createDemoSample = (type: 'single' | 'multi' | 'foreign') => {
    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 600;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Dark slate background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 600, 600);

    const drawGrain = (x: number, y: number, radX: number, radY: number, rot: number, color = '#f1f5f9') => {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rot);
      ctx.beginPath();
      ctx.ellipse(0, 0, radX, radY, 0, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    };

    if (type === 'single') {
      drawGrain(300, 300, 70, 24, Math.PI / 6, '#f8fafc');
    } else if (type === 'multi') {
      // 8 grains in scattered arrangement
      drawGrain(160, 180, 55, 18, 0.3, '#f1f5f9');
      drawGrain(280, 160, 60, 20, -0.4, '#f8fafc');
      drawGrain(420, 200, 50, 17, 0.8, '#fef08a'); // chalky
      drawGrain(190, 320, 58, 19, -0.2, '#fecdd3'); // red grain
      drawGrain(340, 310, 40, 18, 0.5, '#f1f5f9'); // broken
      drawGrain(450, 340, 62, 21, -0.7, '#d6d3d1'); // discoloured
      drawGrain(220, 440, 54, 18, 0.1, '#f1f5f9');
      drawGrain(370, 450, 56, 19, -0.5, '#f1f5f9');
    } else {
      // Rice + foreign matter (stones)
      drawGrain(200, 250, 60, 20, 0.2, '#f8fafc');
      drawGrain(360, 280, 58, 19, -0.4, '#f8fafc');
      // Dark polygon stone
      ctx.fillStyle = '#475569';
      ctx.beginPath();
      ctx.moveTo(300, 400);
      ctx.lineTo(350, 380);
      ctx.lineTo(380, 430);
      ctx.lineTo(310, 440);
      ctx.closePath();
      ctx.fill();
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `demo_${type}_rice.jpg`, { type: 'image/jpeg' });
        setSelectedFile(file);
        setPreviewUrl(canvas.toDataURL('image/jpeg'));
      }
    }, 'image/jpeg');
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Intro Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
          Automated Rice Grain Quality Analysis
        </h1>
        <p className="mt-2 text-base text-slate-400 max-w-2xl mx-auto">
          Upload any photograph of raw milled rice grains (1 grain to thousands, on any background).
          Calculates all 14 official & engineering parameters with standards screening.
        </p>

        {/* Quick Demo Previews */}
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs">
          <span className="text-slate-400 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Quick Demo Samples:
          </span>
          <button
            type="button"
            onClick={() => createDemoSample('single')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md border border-slate-700 transition"
          >
            1 Grain (Single Grain Mode)
          </button>
          <button
            type="button"
            onClick={() => createDemoSample('multi')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md border border-slate-700 transition"
          >
            8 Grains (Multi-Defect Sample)
          </button>
          <button
            type="button"
            onClick={() => createDemoSample('foreign')}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md border border-slate-700 transition"
          >
            Rice + Foreign Matter
          </button>
        </div>
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
              ? 'border-amber-400 bg-amber-500/10 scale-[1.01]'
              : 'border-slate-700 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-900/80'
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
              <div className="relative group max-w-sm rounded-xl overflow-hidden border border-slate-700 bg-slate-950 shadow-2xl">
                <img src={previewUrl} alt="Upload Preview" className="max-h-72 object-contain mx-auto" />
                <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="text-xs text-white bg-slate-800/90 px-3 py-1.5 rounded-lg border border-slate-600">
                    Click or drop to replace
                  </span>
                </div>
              </div>
              <div className="text-xs text-slate-400">
                <span className="font-semibold text-slate-200">{selectedFile?.name}</span> ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 mb-1">
                <Upload className="w-8 h-8" />
              </div>
              <div className="text-sm font-medium text-slate-200">
                Drag and drop your rice image here, or <span className="text-amber-400 underline underline-offset-2">browse</span>
              </div>
              <p className="text-xs text-slate-400 max-w-md">
                Supports JPG, PNG, WEBP, and TIFF. No minimum megapixel restriction. Any lighting or arbitrary background.
              </p>
            </div>
          )}
        </div>

        {/* Options Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          {/* Target Standard Profile */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Standard Grade Reference
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setGrade('grade_a')}
                className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition ${
                  grade === 'grade_a'
                    ? 'border-amber-500 bg-amber-500/10 text-amber-300'
                    : 'border-slate-700 bg-slate-800/60 text-slate-400 hover:text-slate-300'
                }`}
              >
                <div className="font-semibold">Grade A Rice</div>
                <div className="text-[10px] text-slate-400 mt-0.5">India KMS 2026-27 (Admix limit 6.0%)</div>
              </button>
              <button
                type="button"
                onClick={() => setGrade('common')}
                className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition ${
                  grade === 'common'
                    ? 'border-amber-500 bg-amber-500/10 text-amber-300'
                    : 'border-slate-700 bg-slate-800/60 text-slate-400 hover:text-slate-300'
                }`}
              >
                <div className="font-semibold">Common Rice</div>
                <div className="text-[10px] text-slate-400 mt-0.5">India KMS 2026-27 (No admix limit)</div>
              </button>
            </div>
          </div>

          {/* Metric Calibration Toggle */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Metric Calibration
              </label>
              <button
                type="button"
                onClick={() => setEnableManualScale(!enableManualScale)}
                className="text-xs text-amber-400 hover:underline flex items-center gap-1"
              >
                <Sliders className="w-3 h-3" />
                {enableManualScale ? 'Disable Manual Scale' : 'Set Manual Scale (mm)'}
              </button>
            </div>

            {enableManualScale ? (
              <div className="grid grid-cols-2 gap-2 bg-slate-800/60 p-2 rounded-lg border border-slate-700 text-xs">
                <div>
                  <span className="text-[10px] text-slate-400 block mb-1">Known Object Pixels</span>
                  <input
                    type="number"
                    value={refPixels}
                    onChange={(e) => setRefPixels(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-amber-500"
                    placeholder="e.g. 100"
                  />
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 block mb-1">Physical Size (mm)</span>
                  <input
                    type="number"
                    value={refMm}
                    onChange={(e) => setRefMm(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-amber-500"
                    placeholder="e.g. 10"
                  />
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-400 bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/60 flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-medium text-slate-300">Automatic Reference Detection Active:</span>
                  <span className="block text-[11px] text-slate-400 mt-0.5">
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
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm shadow-xl flex items-center justify-center gap-2 transition-all duration-200 ${
              !selectedFile || isLoading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold shadow-amber-500/20 hover:shadow-amber-500/30'
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
