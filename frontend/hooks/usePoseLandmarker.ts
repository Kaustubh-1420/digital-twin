"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LandmarksFilter } from "@/lib/oneEuroFilter";

// Pinned to the installed package version
const WASM_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility: number;
};

export type PoseLandmarks = Landmark[];
export type HandLandmarks = Landmark[];

export function usePoseLandmarker() {
  const [ready, setReady] = useState(false);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ref so bone-driving code can read without React re-renders each frame
  const landmarksRef    = useRef<PoseLandmarks | null>(null);
  // Normalized (image-space) landmarks for the 2D overlay canvas
  const normLandmarksRef = useRef<PoseLandmarks | null>(null);
  // Hand world landmarks (wrist-centered, metric). leftHandRef = user's actual left hand.
  const leftHandRef  = useRef<HandLandmarks | null>(null);
  const rightHandRef = useRef<HandLandmarks | null>(null);

  const landmarkerRef     = useRef<import("@mediapipe/tasks-vision").PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<import("@mediapipe/tasks-vision").HandLandmarker | null>(null);
  const faceLandmarkerRef = useRef<import("@mediapipe/tasks-vision").FaceLandmarker | null>(null);
  // 52 ARKit blendshape scores in MediaPipe canonical order (null when face not detected)
  const faceBlendshapesRef = useRef<number[] | null>(null);
  const videoRef          = useRef<HTMLVideoElement | null>(null);
  const rafRef            = useRef<number>(0);
  const filterRef         = useRef<LandmarksFilter>(new LandmarksFilter(1.0, 0.3));
  const leftHandFilterRef  = useRef<LandmarksFilter>(new LandmarksFilter(1.0, 0.3));
  const rightHandFilterRef = useRef<LandmarksFilter>(new LandmarksFilter(1.0, 0.3));

  // Load the model once on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker } = await import(
          "@mediapipe/tasks-vision"
        );
        const vision = await FilesetResolver.forVisionTasks(WASM_CDN);
        const [lm, hlm, flm] = await Promise.all([
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_URL },
            runningMode: "VIDEO",
            numPoses: 1,
            outputSegmentationMasks: false,
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL },
            runningMode: "VIDEO",
            numHands: 2,
          }),
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: FACE_MODEL_URL },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
          }),
        ]);
        if (!cancelled) {
          landmarkerRef.current     = lm;
          handLandmarkerRef.current = hlm;
          faceLandmarkerRef.current = flm;
          setReady(true);
          console.log("[MediaPipe] pose + hand + face models ready");
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
      console.log("[MediaPipe] webcam started, rAF detection loop running");

      let lastTs = -1;
      let _dbgLmFrame = 0;
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
          normLandmarksRef.current = (result.landmarks?.[0] as PoseLandmarks) ?? null;
          // DEBUG: log first detection and then every 300 frames — remove when stable
          if (raw && ((_dbgLmFrame === 0) || (_dbgLmFrame % 300 === 0))) {
            const h = raw[23]; const s = raw[11];
            console.log(`[MediaPipe] frame=${_dbgLmFrame} worldLandmarks OK len=${raw.length} hipL.y=${h.y.toFixed(3)} shldL.y=${s.y.toFixed(3)} (expect hipL.y<0, shldL.y>0 for Y-up)`);
          }
          if (raw) _dbgLmFrame++;

          // Hand detection — reset each frame then populate from results
          leftHandRef.current = null;
          rightHandRef.current = null;
          const hLm = handLandmarkerRef.current;
          if (hLm) {
            const hResult = hLm.detectForVideo(video, ts);
            hResult.handedness?.forEach((handedness, i) => {
              const rawHand = hResult.worldLandmarks?.[i] as HandLandmarks | undefined;
              if (!rawHand) return;
              // MediaPipe labels hands anatomically ("Right" = user's actual right hand).
              // No swap needed — assign directly.
              const isUserLeft = handedness[0].categoryName === "Left";
              if (isUserLeft) {
                leftHandRef.current = leftHandFilterRef.current.filter(rawHand);
              } else {
                rightHandRef.current = rightHandFilterRef.current.filter(rawHand);
              }
            });
          }

          // Face blendshape detection
          faceBlendshapesRef.current = null;
          const fLm = faceLandmarkerRef.current;
          if (fLm) {
            const fResult = fLm.detectForVideo(video, ts);
            const cats = fResult.faceBlendshapes?.[0]?.categories;
            if (cats && cats.length > 0) {
              faceBlendshapesRef.current = cats.map((c) => c.score);
            }
          }
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
    normLandmarksRef.current = null;
    leftHandRef.current = null;
    rightHandRef.current = null;
    faceBlendshapesRef.current = null;
    filterRef.current = new LandmarksFilter(1.0, 0.3);
    leftHandFilterRef.current  = new LandmarksFilter(1.0, 0.3);
    rightHandFilterRef.current = new LandmarksFilter(1.0, 0.3);
    setActive(false);
  }, []);

  return { ready, active, error, landmarksRef, normLandmarksRef, leftHandRef, rightHandRef, faceBlendshapesRef, videoRef, start, stop };
}
