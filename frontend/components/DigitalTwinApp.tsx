"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import UploadForm from "./UploadForm";
import AvatarViewer from "./AvatarViewer";
import MeasurementsPanel from "./MeasurementsPanel";
import PoseOverlay from "./PoseOverlay";
import { runPipeline, GlbDownloadError } from "@/lib/api";
import { usePoseLandmarker } from "@/hooks/usePoseLandmarker";
import { useServerHealth, type ServerHealth } from "@/hooks/useServerHealth";
import { resetSkeletonDriverState } from "@/lib/poseSolver";

type Status = "idle" | "loading" | "done" | "error";

const HEALTH_UI: Record<ServerHealth, { label: string; color: string; pulse?: boolean }> = {
  checking: { label: "Checking server…", color: "#b5b3ad" },
  ready:    { label: "Server ready", color: "#7ea478" },
  waking:   { label: "Waking server (can take a few minutes)", color: "#d9a441", pulse: true },
  down:     { label: "Server unavailable", color: "#c96a5a" },
  unknown:  { label: "Server status unknown", color: "#b5b3ad" },
};

export default function DigitalTwinApp() {
  const [status, setStatus] = useState<Status>("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<string | null>(null);
  const [glbOnlyFailure, setGlbOnlyFailure] = useState(false);

  const {
    loading: mpLoading, active: webcamActive, error: mpError,
    landmarksRef, normLandmarksRef, leftHandRef, rightHandRef,
    faceBlendshapesRef, videoRef, start: startWebcam, stop: stopWebcam,
  } = usePoseLandmarker();
  const { health, markReady } = useServerHealth();

  const [mirrorMode, setMirrorMode] = useState(false);
  const mirrorRef = useRef(false);

  function toggleMirror() {
    mirrorRef.current = !mirrorRef.current;
    setMirrorMode(mirrorRef.current);
  }

  function handleStopWebcam() {
    stopWebcam();
    resetSkeletonDriverState();
  }

  // runPipeline hands back an object URL now, so the previous one has to be
  // released when it is replaced or discarded.
  function releaseGlbUrl(url: string | null) {
    if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
  }

  async function handleSubmit(file: File, heightCm: number) {
    setStatus("loading");
    setStatusText("Connecting to server…");
    setError(null);
    try {
      setStatusText("Running body estimation (30–60 s on cold GPU)…");
      const result = await runPipeline(file, heightCm, setStatusText);
      setGlbUrl((prev) => {
        releaseGlbUrl(prev);
        return result.glbUrl;
      });
      setMeasurements(result.measurements);
      setStatus("done");
      markReady();
    } catch (e) {
      // Always drop the previous avatar: showing it after a failure is what
      // made a failed run look like a successful one. Measurements survive
      // only when body estimation itself succeeded and just the 3D download
      // failed — they come from this photo, and the stage overlay says the
      // model is missing.
      setGlbUrl((prev) => {
        releaseGlbUrl(prev);
        return null;
      });
      setMeasurements(e instanceof GlbDownloadError ? e.measurements : null);
      setGlbOnlyFailure(e instanceof GlbDownloadError);
      setError(e instanceof Error ? e.message : "Unknown error");
      setStatus("error");
    }
  }

  const loading = status === "loading";
  const hasAvatar = !!glbUrl;

  return (
    <div className="flex flex-col min-h-full md:h-full bg-[#fafaf8] text-[#0c0c0a]">
      {/* Topbar */}
      <header className="flex items-center justify-between gap-2 px-4 md:px-6 h-14 border-b border-black/[0.07] bg-white shrink-0">
        <div className="flex items-center gap-2.5">
          <div
            className="w-[22px] h-[22px] rounded-[7px]"
            style={{
              background: "linear-gradient(135deg, #d9a98d 0%, #b07a5e 100%)",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.4)",
            }}
          />
          <span className="font-semibold text-sm tracking-tight">digital-twin</span>
          <span className="hidden sm:inline text-black/50 text-sm">· studio</span>
        </div>
        <nav className="flex gap-1">
          <Link href="/" className="px-2.5 md:px-3.5 py-1.5 text-xs font-medium rounded-md bg-[#f0eee9] text-[#0c0c0a]">
            Studio
          </Link>
          <Link href="/method" className="px-2.5 md:px-3.5 py-1.5 text-xs font-medium rounded-md text-black/50 hover:text-black/80 transition-colors">
            Method
          </Link>
          <a
            href="https://github.com/Kaustubh-1420/digital-twin"
            target="_blank"
            rel="noopener noreferrer"
            className="px-2.5 md:px-3.5 py-1.5 text-xs font-medium rounded-md text-black/50 hover:text-black/80 transition-colors"
          >
            GitHub ↗
          </a>
        </nav>
        <div className="flex items-center gap-1.5 text-xs text-black/55 shrink-0">
          <span
            className={`w-[7px] h-[7px] rounded-full shrink-0 ${HEALTH_UI[health].pulse ? "animate-pulse" : ""}`}
            style={{
              backgroundColor: HEALTH_UI[health].color,
              boxShadow: `0 0 0 3px ${HEALTH_UI[health].color}2e`,
            }}
          />
          <span role="status" className="sr-only md:not-sr-only">{HEALTH_UI[health].label}</span>
        </div>
      </header>

      {/* Main 3-column body */}
      <div className="flex flex-col md:flex-row flex-1 md:min-h-0">
        {/* LEFT — controls */}
        <aside className="w-full md:w-[340px] shrink-0 flex flex-col gap-3 p-4 border-b md:border-b-0 md:border-r border-black/[0.07] bg-white md:overflow-y-auto">
          <div>
            <h1 className="font-serif font-normal text-[22px] leading-[1.15] tracking-tight">
              Your body,<br />in real time.
            </h1>
            <p className="text-xs text-black/60 mt-1.5 leading-relaxed">
              One photo. Six measurements. A skinned avatar that mirrors you live through your webcam.
            </p>
            <p className="md:hidden text-[11px] text-black/45 mt-1.5">Best experienced on desktop.</p>
          </div>

          {/* pipeline pills */}
          <div className="flex flex-wrap gap-1.5">
            {["PyMAF-X", "SMPL-X", "MediaPipe", "Three.js"].map((p) => (
              <span
                key={p}
                className="px-2.5 py-1 border border-black/10 rounded-full text-[11px] font-mono text-black/65 bg-[#fafaf8]"
              >
                {p}
              </span>
            ))}
          </div>

          <UploadForm
            onSubmit={handleSubmit}
            loading={loading}
            status={statusText}
            error={error}
            hasAvatar={hasAvatar}
          />

          {/* Webcam controls — only after avatar loaded */}
          {hasAvatar && (
            <div className="flex flex-col gap-2 border-t border-black/[0.07] pt-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium uppercase tracking-wider text-black/50">
                  Real-time mirror
                </span>
                {webcamActive && (
                  <span className="flex items-center gap-1.5 text-xs text-[#7ea478]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#7ea478] animate-pulse" />
                    Tracking
                  </span>
                )}
              </div>

              <button
                onClick={webcamActive ? handleStopWebcam : startWebcam}
                disabled={mpLoading}
                className="w-full py-2.5 rounded-lg font-medium text-sm transition-colors
                  bg-[#0c0c0a] hover:bg-black disabled:opacity-40 disabled:cursor-not-allowed text-[#fafaf8]"
              >
                {mpLoading ? "Loading pose models (first time only)…" : webcamActive ? "⏹ Stop webcam" : "▶ Start webcam mirror"}
              </button>

              <button
                onClick={toggleMirror}
                className="w-full py-2 rounded-lg text-sm transition-colors
                  border border-black/10 hover:border-black/30 text-black/55 hover:text-black/90"
              >
                {mirrorMode ? "Mirror: ON" : "Mirror: OFF"}
              </button>

              {mpError && <p className="text-xs text-red-500">{mpError}</p>}
            </div>
          )}
        </aside>

        {/* CENTER — viewer stage */}
        <main className="order-first md:order-none h-[60vh] min-h-[360px] shrink-0 md:h-auto md:min-h-0 md:shrink md:flex-1 min-w-0 relative dt-stage">
          {/* stage label */}
          <div className="absolute top-5 left-6 z-10 pointer-events-none">
            <div className="font-serif italic text-sm text-black/55">Studio · Plate 01</div>
          </div>
          <div className="absolute top-5 right-6 z-10 pointer-events-none font-mono text-[10px] text-black/50 text-right leading-relaxed tracking-wider">
            10,475 verts · 55 joints<br />100 morph targets
          </div>

          <div className="absolute left-6 right-6 top-14 bottom-8">
            <AvatarViewer
              glbUrl={glbUrl}
              loading={loading}
              landmarksRef={landmarksRef}
              normLandmarksRef={normLandmarksRef}
              leftHandRef={leftHandRef}
              rightHandRef={rightHandRef}
              faceBlendshapesRef={faceBlendshapesRef}
              mirrorRef={mirrorRef}
              active={webcamActive}
            />
            <PoseOverlay
              videoRef={videoRef}
              normLandmarksRef={normLandmarksRef}
              active={webcamActive}
            />

            {/* The stage falls back to the showcase mannequin whenever there is
                no avatar. After a failed run that reads as a result, so cover
                it and say plainly that nothing was produced. */}
            {status === "error" && (
              <div className="absolute inset-0 z-20 flex items-center justify-center rounded-2xl bg-[#fafaf8]/95 backdrop-blur-sm">
                <div className="max-w-[300px] text-center px-6">
                  <div className="font-serif italic text-sm text-black/55 mb-2">
                    {glbOnlyFailure ? "3D model couldn't be loaded" : "No avatar generated"}
                  </div>
                  <p className="text-xs text-black/60 leading-relaxed">
                    {error}
                  </p>
                  <p className="text-[11px] text-black/40 leading-relaxed mt-3">
                    {glbOnlyFailure
                      ? "Your measurements were estimated and are shown in the panel. Only the 3D view is missing."
                      : "The figure shown before this is a sample model, not your result."}
                  </p>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* RIGHT — measurements */}
        <aside className="w-full md:w-[280px] shrink-0 flex flex-col gap-5 p-6 border-t md:border-t-0 md:border-l border-black/[0.07] bg-white md:overflow-y-auto">
          <MeasurementsPanel text={measurements} />
        </aside>
      </div>
    </div>
  );
}
