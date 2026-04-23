import { NextRequest } from "next/server";

export async function GET(request: NextRequest) {
  const url = request.nextUrl.searchParams.get("url");
  if (!url) return new Response("Missing url param", { status: 400 });

  // Only allow HF Space URLs to prevent open-proxy abuse
  if (!url.startsWith("https://") || !url.includes(".hf.space/")) {
    return new Response("Forbidden", { status: 403 });
  }

  const upstream = await fetch(url);
  if (!upstream.ok) {
    return new Response("Upstream error", { status: upstream.status });
  }

  const body = await upstream.arrayBuffer();
  return new Response(body, {
    headers: {
      "Content-Type": "model/gltf-binary",
      "Cache-Control": "no-store",
    },
  });
}
