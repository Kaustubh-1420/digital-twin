"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";

type Props = {
  onSubmit: (file: File, heightCm: number) => void;
  loading: boolean;
  status: string;
  error: string | null;
};

export default function UploadForm({ onSubmit, loading, status, error }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heightCm, setHeightCm] = useState(170);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function acceptFile(f: File) {
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) acceptFile(f);
  }

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) acceptFile(f);
  }

  function handleSubmit() {
    if (file) onSubmit(file, heightCm);
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Upload area */}
      <div
        className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors cursor-pointer
          ${dragging ? "border-indigo-400 bg-indigo-950/30" : "border-zinc-700 hover:border-zinc-500"}
          ${preview ? "h-52" : "h-44"}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onFileChange}
        />
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Preview"
            className="h-full w-full object-contain rounded-xl p-1"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-zinc-400 select-none">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span className="text-sm">Drop photo or click to upload</span>
            <span className="text-xs text-zinc-600">Full-body, front-facing works best</span>
          </div>
        )}
      </div>

      {/* Height slider */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between text-sm text-zinc-400">
          <span>Height</span>
          <span className="text-zinc-100 font-medium tabular-nums">{heightCm} cm</span>
        </div>
        <input
          type="range"
          min={100}
          max={250}
          value={heightCm}
          onChange={(e) => setHeightCm(Number(e.target.value))}
          className="w-full accent-indigo-500"
        />
        <div className="flex justify-between text-xs text-zinc-600">
          <span>100 cm</span>
          <span>250 cm</span>
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className="w-full py-3 rounded-lg font-medium text-sm transition-colors
          bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-600
          disabled:cursor-not-allowed text-white"
      >
        {loading ? "Generating…" : "Generate Avatar →"}
      </button>

      {/* Status / error */}
      {loading && (
        <p className="text-xs text-zinc-400 text-center">
          {status || "Warming up GPU, first run takes ~30 s…"}
        </p>
      )}
      {error && (
        <p className="text-xs text-red-400 text-center">{error}</p>
      )}
    </div>
  );
}
