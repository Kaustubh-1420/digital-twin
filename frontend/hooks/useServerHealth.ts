"use client";

import { useCallback, useEffect, useState } from "react";
import type { HealthState } from "@/app/api/health/route";

const POLL_MS = 10_000;
const MAX_POLL_MS = 5 * 60_000;

export type ServerHealth = HealthState | "checking";

/**
 * Polls /api/health on mount and keeps polling while the Space is waking, so
 * the header shows the backend's real state instead of a hardcoded "ready".
 * The first request also wakes a sleeping Space (see the route).
 */
export function useServerHealth() {
  const [health, setHealth] = useState<ServerHealth>("checking");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const startedAt = Date.now();

    const poll = async () => {
      let state: HealthState = "unknown";
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (res.ok) state = (await res.json()).state;
      } catch {
        // stays "unknown"
      }
      if (cancelled) return;
      // A successful generation may have already proven the server is up.
      setHealth((prev) => (prev === "ready" && state !== "down" ? prev : state));
      if (state === "waking" && Date.now() - startedAt < MAX_POLL_MS) {
        timer = setTimeout(poll, POLL_MS);
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  /** A generation that succeeded is the strongest health signal there is. */
  const markReady = useCallback(() => setHealth("ready"), []);

  return { health, markReady };
}
