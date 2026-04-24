import { Client } from "@gradio/client";

export type PipelineResult = {
  glbUrl: string;
  measurements: string;
};

export async function runPipeline(
  imageFile: File,
  heightCm: number
): Promise<PipelineResult> {
  const client = await Client.connect("Kaustubh1420/digital-twin");
  // fn_index 0 = run_pipeline (first click handler in the Blocks app)
  const result = await client.predict(0, [imageFile, heightCm]);
  console.log("[api] raw result.data:", JSON.stringify(result.data));
  const [glbData, measurementsText, statusText] = result.data as [
    { url: string } | null,
    string,
    string
  ];

  if (!glbData?.url) {
    throw new Error(statusText || "No avatar returned from server.");
  }

  // Proxy through local route to avoid cross-origin issues with the HF Space URL
  const proxiedUrl = `/api/glb?url=${encodeURIComponent(glbData.url)}`;
  return { glbUrl: proxiedUrl, measurements: measurementsText };
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
