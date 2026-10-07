export type PipelineResult = {
  glbUrl: string;
  measurements: string;
};

/**
 * Body estimation succeeded but the 3D file never arrived. Carries the
 * measurements, which came from the user's actual photo and are still valid —
 * only the avatar is missing.
 */
export class GlbDownloadError extends Error {
  readonly measurements: string;

  constructor(message: string, measurements: string) {
    super(message);
    this.name = "GlbDownloadError";
    this.measurements = measurements;
  }
}

const GLB_ATTEMPTS = 3;
const GLB_BACKOFF_MS = 1500;

/**
 * Download the generated GLB, retrying on transient upstream failures.
 *
 * The 503 seen on this path is transient, not permanent: the same URL that
 * fails on first request serves a valid 13MB file seconds later. Retrying
 * recovers it. Returning a Blob (rather than letting GLTFLoader fetch the
 * URL a second time) also means the file crosses the wire exactly once, and
 * guarantees the caller cannot report success for a file that never arrived.
 */
async function fetchGlb(proxiedUrl: string): Promise<Blob> {
  let lastError = "unknown error";

  for (let attempt = 1; attempt <= GLB_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(proxiedUrl);
      if (res.ok) {
        const blob = await res.blob();
        if (blob.size === 0) throw new Error("empty file");
        return blob;
      }
      lastError = `HTTP ${res.status}`;
      // 4xx (other than timeout/rate-limit) will not resolve on retry.
      if (res.status < 500 && res.status !== 408 && res.status !== 429) break;
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e);
    }

    if (attempt < GLB_ATTEMPTS) {
      await new Promise((r) => setTimeout(r, GLB_BACKOFF_MS * attempt));
    }
  }

  throw new Error(
    `Avatar was generated but could not be downloaded (${lastError}). ` +
      `Please try again.`
  );
}

const PREDICT_ATTEMPTS = 2;
const PREDICT_RETRY_MS = 2000;

type PredictBody = { glbUrl?: string; measurements?: string; error?: string; retryable?: boolean };

/**
 * One /api/predict call. Read as text and parsed defensively: a platform-level
 * failure (function timeout, crash, cold-start error) returns an HTML error
 * page rather than JSON, and res.json() throws on that before res.ok is ever
 * checked — which surfaced to the user as `Unexpected token 'A'` instead of
 * something they could act on.
 */
async function callPredict(formData: FormData) {
  const res = await fetch("/api/predict", { method: "POST", body: formData });
  const raw = await res.text();
  let data: PredictBody = {};
  let parsed = false;
  try {
    data = JSON.parse(raw);
    parsed = true;
  } catch {
    // Left empty: handled by the status checks in the caller.
  }
  return { res, data, parsed };
}

/** Failures that look like a cold or restarting backend rather than an answer
 *  about this photo: gateway errors, the platform's own (non-JSON) error page,
 *  or our route flagging an infrastructure exception. */
function looksTransient(status: number, data: PredictBody, parsed: boolean) {
  return status === 503 || status === 504 || (status >= 500 && !parsed) || data.retryable === true;
}

export async function runPipeline(
  imageFile: File,
  heightCm: number,
  onStatus?: (text: string) => void
): Promise<PipelineResult> {
  // Routed through our own /api/predict (server-side) instead of connecting to
  // the HF Space directly from the browser, so HF_TOKEN never reaches the
  // client. (Originally forced by CORS: @gradio/client < 2.7 hardcoded
  // `credentials: "include"`, which HF Spaces' wildcard CORS rejects.)
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("heightCm", String(heightCm));

  let { res, data, parsed } = await callPredict(formData);

  // A sleeping GPU often fails the first request and serves the second.
  for (let attempt = 2; attempt <= PREDICT_ATTEMPTS && !res.ok && looksTransient(res.status, data, parsed); attempt++) {
    onStatus?.("GPU was asleep, retrying…");
    await new Promise((r) => setTimeout(r, PREDICT_RETRY_MS));
    ({ res, data, parsed } = await callPredict(formData));
  }

  if (!res.ok) {
    throw new Error(
      data.error || `Body estimation failed (HTTP ${res.status}). Please try again.`
    );
  }

  if (!data.glbUrl || typeof data.measurements !== "string") {
    throw new Error(
      "Body estimation returned an unexpected response. Please try again."
    );
  }

  // Proxy through local route to avoid cross-origin issues with the HF Space URL
  const proxiedUrl = `/api/glb?url=${encodeURIComponent(data.glbUrl)}`;

  // Fetch it here rather than handing the URL straight to the viewer. If this
  // throws, the caller surfaces a real error instead of rendering the stock
  // mannequin next to measurements that came from the user's actual photo.
  let blob: Blob;
  try {
    blob = await fetchGlb(proxiedUrl);
  } catch (e) {
    throw new GlbDownloadError(
      e instanceof Error ? e.message : String(e),
      data.measurements
    );
  }

  return {
    glbUrl: URL.createObjectURL(blob),
    measurements: data.measurements,
  };
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
