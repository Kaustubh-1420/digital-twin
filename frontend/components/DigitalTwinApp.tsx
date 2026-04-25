"use client";

import { useState, useRef } from "react";
import UploadForm from "./UploadForm";
import AvatarViewer from "./AvatarViewer";
import MeasurementsPanel from "./MeasurementsPanel";
import PoseOverlay from "./PoseOverlay";
import { runPipeline } from "@/lib/api";
import { usePoseLandmarker } from "@/hooks/usePoseLandmarker";
import { resetSkeletonDriverState } from "@/lib/poseSolver";

type Status = "idle" | "loading" | "done" | "error";

export default function DigitalTwinApp() {
  const [status, setStatus] = useState<Status>("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<string | null>(null);

  const { ready: mpReady, active: webcamActive, error: mpError, landmarksRef, normLandmarksRef, videoRef, start: startWebcam, stop: stopWebcam } =
    usePoseLandmarker();

  const [mirrorMode, setMirrorMode] = useState(false);
  const mirrorRef = useRef(false);

  function toggleMirror() {
    mirrorRef.current = !mirrorRef.current;
    setMirrorMode(mirrorRef.current);
    console.log(`[App] mirror toggled → ${mirrorRef.current ? "ON" : "OFF"}`);
  }

  function handleStopWebcam() {
    stopWebcam();
    resetSkeletonDriverState();
    console.log("[App] webcam stopped, skeleton state reset");
  }

  async function handleSubmit(file: File, heightCm: number) {
    setStatus("loading");
    setStatusText("Connecting to server…");
    setError(null);
    console.log(`[App] submit: file=${file.name} (${(file.size / 1024).toFixed(0)} KB), height=${heightCm} cm`);

    try {
      setStatusText("Running body estimation (30–60 s on cold GPU)…");
      const result = await runPipeline(file, heightCm);
      setGlbUrl(result.glbUrl);
      setMeasurements(result.measurements);
      setStatus("done");
      console.log("[App] pipeline done — glbUrl=", result.glbUrl, "measurements=", result.measurements);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setStatus("error");
      console.error("[App] pipeline error:", e);
    }
  }

  const loading = status === "loading";
  const hasAvatar = !!glbUrl;

  return (
    <div className="flex h-full">
      {/* Left panel */}
      <aside className="w-80 shrink-0 flex flex-col gap-6 p-6 border-r border-zinc-800 overflow-y-auto">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">digital-twin</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Photo → personalized SMPL-X avatar → real-time mirroring
          </p>
        </div>

        <UploadForm
          onSubmit={handleSubmit}
          loading={loading}
          status={statusText}
          error={error}
        />

        {/* Webcam controls — only shown once an avatar exists */}
        {hasAvatar && (
          <div className="flex flex-col gap-2 border-t border-zinc-800 pt-5">
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">Real-time mirroring</span>
              {webcamActive && (
                <span className="flex items-center gap-1.5 text-xs text-green-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  Tracking
                </span>
              )}
            </div>

            <button
              onClick={webcamActive ? handleStopWebcam : startWebcam}
              disabled={!mpReady}
              className="w-full py-3 rounded-lg font-medium text-sm transition-colors
                bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed text-white"
            >
              {!mpReady
                ? "Loading pose model…"
                : webcamActive
                ? "⏹ Stop webcam"
                : "▶ Start webcam mirror"}
            </button>

            {/* Mirror mode toggle */}
            <button
              onClick={toggleMirror}
              className="w-full py-2 rounded-lg text-sm transition-colors
                border border-zinc-700 hover:border-zinc-500 text-zinc-400 hover:text-zinc-200"
            >
              {mirrorMode ? "Mirror: ON" : "Mirror: OFF"}
            </button>

            {mpError && (
              <p className="text-xs text-red-400">{mpError}</p>
            )}
          </div>
        )}
      </aside>

      {/* Right panel */}
      <main className="flex flex-1 min-w-0 min-h-0 p-6 gap-6">
        <div className="relative flex flex-col flex-1 min-w-0 min-h-0">
          <AvatarViewer
            glbUrl={glbUrl}
            loading={loading}
            landmarksRef={landmarksRef}
            normLandmarksRef={normLandmarksRef}
            mirrorRef={mirrorRef}
            active={webcamActive}
          />
          <PoseOverlay
            videoRef={videoRef}
            normLandmarksRef={normLandmarksRef}
            active={webcamActive}
          />
        </div>

        <aside className="w-56 shrink-0 flex flex-col gap-4 overflow-y-auto">
          <MeasurementsPanel text={measurements} />
        </aside>
      </main>
    </div>
  );
}
