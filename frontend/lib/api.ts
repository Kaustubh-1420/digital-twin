export type PipelineResult = {
  glbUrl: string;
  measurements: string;
};

export async function runPipeline(
  imageFile: File,
  heightCm: number
): Promise<PipelineResult> {
  // Routed through our own /api/predict (server-side) instead of connecting to
  // the HF Space directly from the browser — @gradio/client hardcodes
  // `credentials: "include"` on its fetches, which HF Spaces' wildcard CORS
  // headers reject outright.
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("heightCm", String(heightCm));

  const res = await fetch("/api/predict", { method: "POST", body: formData });
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || "Pipeline failed.");
  }

  // Proxy through local route to avoid cross-origin issues with the HF Space URL
  const proxiedUrl = `/api/glb?url=${encodeURIComponent(data.glbUrl)}`;
  return { glbUrl: proxiedUrl, measurements: data.measurements };
}

export function parseMeasurements(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    // "Chest              92.3 cm"
    const match = trimmed.match(/^(.+?)\s{2,}([\d.]+\s*cm)$/);
    if (match) out[match[1].trim()] = match[2].trim();
  }
  return out;
}
