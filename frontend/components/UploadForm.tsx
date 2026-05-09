"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";

type Props = {
  onSubmit: (file: File, heightCm: number) => void;
  loading: boolean;
  status: string;
  error: string | null;
  hasAvatar?: boolean;
};

export default function UploadForm({ onSubmit, loading, status, error, hasAvatar }: Props) {
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

  async function loadSample(path: string) {
    const res = await fetch(path);
    const blob = await res.blob();
    const ext = path.split(".").pop() ?? "jpg";
    const f = new File([blob], `sample.${ext}`, { type: blob.type });
    acceptFile(f);
  }

  function handleSubmit() {
    if (file) onSubmit(file, heightCm);
  }

  const heightPct = ((heightCm - 100) / 150) * 100;

  return (
    <div className="flex flex-col gap-4">
      {/* Sample photos — hidden after avatar generated */}
      {!hasAvatar && (
        <div className="flex flex-col gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-black/50">
            Try a sample
          </span>
          <div className="grid grid-cols-2 gap-2">
            {[
              { src: "/samples/sample-female.jpg", label: "Female" },
              { src: "/samples/sample-male.jpg", label: "Male" },
            ].map(({ src, label }) => (
              // eslint-disable-next-line @next/next/no-img-element
              <div
                key={src}
                onClick={() => loadSample(src)}
                className="relative aspect-[3/4] rounded-[10px] overflow-hidden bg-[#f0eee9] border border-black/[0.06] cursor-pointer hover:border-black/30 transition-colors"
              >
                <img
                  src={src}
                  alt={label}
                  className="absolute inset-0 w-full h-full object-cover object-top"
                />
                <span className="absolute top-2 left-2 text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-white/85 text-black/70 backdrop-blur-sm">
                  {label}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload area — compact row after avatar generated, full drop zone before */}
      {hasAvatar ? (
        <div
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-black/[0.08] bg-[#fafaf8] cursor-pointer hover:border-black/20 transition-colors"
        >
          {preview && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="Preview" className="w-8 h-10 object-cover rounded-md shrink-0" />
          )}
          <span className="text-[12px] text-black/55 flex-1 truncate">
            {file?.name ?? "No file selected"}
          </span>
          <span className="text-[11px] text-black/35 shrink-0">Change</span>
          <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
        </div>
      ) : (
        <div
          className={`relative flex flex-col items-center justify-center rounded-xl border-[1.5px] border-dashed transition-colors cursor-pointer text-center
            ${dragging ? "border-black/40 bg-black/[0.03]" : "border-black/[0.18] hover:border-black/30 bg-[#fafaf8]"}
            ${preview ? "h-44" : "py-5 px-5"}`}
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
            <>
              <div className="w-9 h-9 rounded-[10px] bg-white border border-black/[0.08] flex items-center justify-center mb-2">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="text-black/60">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <span className="text-[13px] font-medium">Drop photo or click to upload</span>
              <span className="text-[11px] text-black/50 mt-0.5">Full-body, front-facing works best</span>
            </>
          )}
        </div>
      )}

      {/* Height slider */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-baseline">
          <span className="text-xs font-medium text-black/70">Height</span>
          <span className="text-sm font-medium tabular-nums">
            {heightCm} <span className="text-black/40 font-normal">cm</span>
          </span>
        </div>
        <div className="relative h-1 bg-black/[0.08] rounded-full">
          <div
            className="absolute left-0 top-0 h-full bg-[#b07a5e] rounded-full"
            style={{ width: `${heightPct}%` }}
          />
          <input
            type="range"
            min={100}
            max={250}
            value={heightCm}
            onChange={(e) => setHeightCm(Number(e.target.value))}
            className="absolute inset-0 w-full opacity-0 cursor-pointer"
          />
          <div
            className="absolute top-1/2 w-[18px] h-[18px] rounded-full bg-white border border-black/[0.12] pointer-events-none"
            style={{
              left: `${heightPct}%`,
              transform: "translate(-50%, -50%)",
              boxShadow: "0 2px 6px rgba(12,12,10,0.12)",
            }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-black/40 tabular-nums">
          <span>100 cm</span>
          <span>250 cm</span>
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        className="w-full py-2.5 rounded-[10px] font-medium text-[13px] transition-colors flex justify-between items-center px-4
          bg-[#0c0c0a] hover:bg-black disabled:bg-black/[0.05] disabled:text-black/30
          disabled:cursor-not-allowed text-[#fafaf8]"
      >
        <span>{loading ? "Generating…" : hasAvatar ? "Re-generate" : "Generate avatar"}</span>
        <span>→</span>
      </button>

      {loading && (
        <p className="text-xs text-black/50 text-center">
          {status || "Warming up GPU, first run takes ~30 s…"}
        </p>
      )}
      {error && (
        <p className="text-xs text-red-500 text-center">{error}</p>
      )}
    </div>
  );
}
