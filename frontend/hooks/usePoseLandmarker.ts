"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LandmarksFilter } from "@/lib/oneEuroFilter";

// Pinned to the installed package version
const WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility: number;
};

export type PoseLandmarks = Landmark[];

export function usePoseLandmarker() {
  const [ready, setReady] = useState(false);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ref so bone-driving code can read without React re-renders each frame
  const landmarksRef = useRef<PoseLandmarks | null>(null);

  const landmarkerRef = useRef<import("@mediapipe/tasks-vision").PoseLandmarker | null>(null);
  const videoRef      = useRef<HTMLVideoElement | null>(null);
  const rafRef        = useRef<number>(0);
  const filterRef     = useRef<LandmarksFilter>(new LandmarksFilter(1.0, 0.3));

  // Load the model once on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { FilesetResolver, PoseLandmarker } = await import(
          "@mediapipe/tasks-vision"
        );
        const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
        const lm = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL },
          runningMode: "VIDEO",
          numPoses: 1,
          outputSegmentationMasks: false,
        });
        if (!cancelled) {
          landmarkerRef.current = lm;
          setReady(true);
        }
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "MediaPipe load failed");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const start = useCallback(async () => {
    if (!landmarkerRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.playsInline = true;
      video.muted = true;
      await video.play();
      videoRef.current = video;
      setActive(true);

      let lastTs = -1;
      const detect = () => {
        const video = videoRef.current;
        const lm = landmarkerRef.current;
        if (!video || !lm) return;

        const ts = performance.now();
        if (ts !== lastTs && video.readyState >= 2) {
          // detectForVideo requires strictly increasing timestamps
          lastTs = ts;
          const result = lm.detectForVideo(video, ts);
          const raw = result.worldLandmarks?.[0];
          landmarksRef.current = raw ? filterRef.current.filter(raw) : null;
        }
        rafRef.current = requestAnimationFrame(detect);
      };
      rafRef.current = requestAnimationFrame(detect);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Webcam access denied");
    }
  }, []);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream)
        .getTracks()
        .forEach((t) => t.stop());
    }
    videoRef.current = null;
    landmarksRef.current = null;
    filterRef.current = new LandmarksFilter(1.0, 0.3); // reset filter state
    setActive(false);
  }, []);

  return { ready, active, error, landmarksRef, start, stop };
}
