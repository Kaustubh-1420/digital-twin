import { NextRequest, NextResponse } from "next/server";
import { Client } from "@gradio/client";

// ZeroGPU cold start can take 30-60s
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const formData = await request.formData();
  const file = formData.get("image");
  const heightCm = Number(formData.get("heightCm"));

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "Missing image file" }, { status: 400 });
  }

  try {
    const client = await Client.connect("Kaustubh1420/digital-twin");
    // fn_index 0 = run_pipeline (first click handler in the Blocks app)
    const result = await client.predict(0, [file, heightCm]);
    const [glbData, measurementsText, statusText] = result.data as [
      { url: string } | null,
      string,
      string
    ];

    if (!glbData?.url) {
      return NextResponse.json(
        { error: statusText || "No avatar returned from server." },
        { status: 502 }
      );
    }

    return NextResponse.json({
      glbUrl: glbData.url,
      measurements: measurementsText,
    });
  } catch (e) {
    // An exception here is infrastructure (connect failed, Space restarting,
    // GPU allocation), unlike the "no avatar" 502 above, which is the app's
    // own answer for this photo. Only this kind is worth retrying.
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Pipeline failed.", retryable: true },
      { status: 502 }
    );
  }
}
