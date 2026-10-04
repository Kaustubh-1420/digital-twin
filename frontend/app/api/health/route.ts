import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const RUNTIME_URL = "https://huggingface.co/api/spaces/Kaustubh1420/digital-twin/runtime";
const SPACE_URL = "https://kaustubh1420-digital-twin.hf.space";

export type HealthState = "ready" | "waking" | "down" | "unknown";

// HF Space stages that resolve on their own if we wait.
const STARTING = new Set([
  "SLEEPING",
  "BUILDING",
  "RUNNING_BUILDING",
  "APP_STARTING",
  "RUNNING_APP_STARTING",
]);

async function get(url: string, timeoutMs: number) {
  return fetch(url, { cache: "no-store", signal: AbortSignal.timeout(timeoutMs) });
}

/**
 * Real backend status for the header dot, replacing a hardcoded "Server ready".
 * Reads the Space's public runtime stage; when it is asleep, also requests the
 * app itself, which is what wakes a sleeping Space — so the warm-up starts as
 * soon as someone opens the page, not when they press Generate.
 */
export async function GET() {
  let stage: string;
  try {
    const res = await get(RUNTIME_URL, 5000);
    if (!res.ok) return NextResponse.json({ state: "unknown" satisfies HealthState });
    stage = (await res.json()).stage;
  } catch {
    return NextResponse.json({ state: "unknown" satisfies HealthState });
  }

  if (stage === "RUNNING") {
    // The container can be RUNNING before Gradio answers; confirm the app does.
    try {
      const res = await get(`${SPACE_URL}/config`, 5000);
      return NextResponse.json({ state: (res.ok ? "ready" : "waking") satisfies HealthState, stage });
    } catch {
      return NextResponse.json({ state: "waking" satisfies HealthState, stage });
    }
  }

  if (STARTING.has(stage)) {
    if (stage === "SLEEPING") {
      // Awaited, not fire-and-forget: a serverless function can be frozen the
      // moment it responds, before an un-awaited request ever leaves. The
      // timeout only bounds our wait; the wake-up is triggered on arrival.
      await get(SPACE_URL, 3000).catch(() => {});
    }
    return NextResponse.json({ state: "waking" satisfies HealthState, stage });
  }

  // PAUSED, STOPPED, RUNTIME_ERROR, BUILD_ERROR, CONFIG_ERROR, NO_APP_FILE…
  return NextResponse.json({ state: "down" satisfies HealthState, stage });
}
