import { NextRequest } from "next/server";

// The first read of a freshly generated GLB from the Space is slow — measured
// at ~18s for a 13MB file, because gradio is serving it off disk cold. The
// Next.js default cap of 10s was cutting that off mid-flight, which surfaced
// to the browser as a 503 on the payload path. /api/predict already carries
// the same allowance for the same reason.
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  const url = request.nextUrl.searchParams.get("url");
  if (!url) return new Response("Missing url param", { status: 400 });

  // Only allow our specific HF Space to prevent open-proxy abuse
  if (!url.startsWith("https://kaustubh1420-digital-twin.hf.space/")) {
    return new Response("Forbidden", { status: 403 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(url);
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    return new Response(`Upstream fetch failed: ${detail}`, { status: 502 });
  }

  if (!upstream.ok || !upstream.body) {
    return new Response(`Upstream error ${upstream.status}`, {
      status: upstream.ok ? 502 : upstream.status,
    });
  }

  // Stream straight through instead of buffering. The previous
  // `await upstream.arrayBuffer()` held all 13MB in function memory and
  // produced a ~4.37MB gzipped response — within 3% of Vercel's 4.5MB
  // serverless response ceiling, so a slightly larger avatar would have
  // failed outright. Streaming has no such cap and starts delivery sooner.
  // Content-Length is deliberately omitted: fetch() transparently decodes
  // upstream gzip, so the upstream value would not match what we emit.
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "model/gltf-binary",
      "Cache-Control": "no-store",
    },
  });
}
