"use client";

import { useState } from "react";
import UploadForm from "./UploadForm";
import AvatarViewer from "./AvatarViewer";
import MeasurementsPanel from "./MeasurementsPanel";
import { runPipeline } from "@/lib/api";

type Status = "idle" | "loading" | "done" | "error";

export default function DigitalTwinApp() {
  const [status, setStatus] = useState<Status>("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [glbUrl, setGlbUrl] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<string | null>(null);

  async function handleSubmit(file: File, heightCm: number) {
    setStatus("loading");
    setStatusText("Connecting to server…");
    setError(null);

    try {
      setStatusText("Running body estimation (30–60 s on cold GPU)…");
      const result = await runPipeline(file, heightCm);
      setGlbUrl(result.glbUrl);
      setMeasurements(result.measurements);
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setStatus("error");
    }
  }

  const loading = status === "loading";

  return (
    <div className="flex h-full">
      {/* Left panel — upload form */}
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
      </aside>

      {/* Right panel — viewer + measurements */}
      <main className="flex flex-1 min-w-0 min-h-0 p-6 gap-6">
        {/* 3D viewer fills available height */}
        <div className="flex flex-col flex-1 min-w-0 min-h-0">
          <AvatarViewer glbUrl={glbUrl} loading={loading} />
        </div>

        {/* Measurements sidebar */}
        <aside className="w-56 shrink-0 flex flex-col gap-4 overflow-y-auto">
          <MeasurementsPanel text={measurements} />
        </aside>
      </main>
    </div>
  );
}
