"use client";

import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

// Connections to draw — [from, to] landmark index pairs
const CONNECTIONS: [number, number][] = [
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15],
  // Right arm
  [12, 14], [14, 16],
  // Left leg
  [23, 25], [25, 27], [27, 31],
  // Right leg
  [24, 26], [26, 28], [28, 32],
];

const VISIBILITY_THRESHOLD = 0.5;

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  active: boolean;
};

export default function PoseOverlay({ videoRef, normLandmarksRef, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number>(0);

  useEffect(() => {
    if (!active) return;

    const draw = () => {
      const canvas = canvasRef.current;
      const video  = videoRef.current;
      const lms    = normLandmarksRef.current;
      if (!canvas) { rafRef.current = requestAnimationFrame(draw); return; }

      const ctx = canvas.getContext("2d");
      if (!ctx) { rafRef.current = requestAnimationFrame(draw); return; }

      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);

      // Draw mirrored video frame
      if (video && video.readyState >= 2) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -W, 0, W, H);
        ctx.restore();
      }

      if (lms && lms.length >= 33) {
        // Landmarks are in [0,1] image space; mirror X to match the flipped video
        const px = (lm: PoseLandmarks[number]) => (1 - lm.x) * W;
        const py = (lm: PoseLandmarks[number]) => lm.y * H;
        const ok = (i: number) =>
          (lms[i]?.visibility ?? 1) >= VISIBILITY_THRESHOLD;

        // Connections
        ctx.strokeStyle = "rgba(99, 102, 241, 0.85)";
        ctx.lineWidth = 2;
        for (const [a, b] of CONNECTIONS) {
          if (!ok(a) || !ok(b)) continue;
          ctx.beginPath();
          ctx.moveTo(px(lms[a]), py(lms[a]));
          ctx.lineTo(px(lms[b]), py(lms[b]));
          ctx.stroke();
        }

        // Joints
        for (let i = 0; i < lms.length; i++) {
          if (!ok(i)) continue;
          ctx.beginPath();
          ctx.arc(px(lms[i]), py(lms[i]), 3, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(167, 139, 250, 0.9)";
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, videoRef, normLandmarksRef]);

  if (!active) return null;

  return (
    <div className="absolute bottom-3 right-3 rounded-lg overflow-hidden border border-zinc-700 shadow-xl"
         style={{ width: 200, height: 150 }}>
      <canvas
        ref={canvasRef}
        width={200}
        height={150}
        className="block w-full h-full"
      />
      <div className="absolute top-1 left-1.5 text-[10px] text-zinc-400 font-mono leading-none">
        pose cam
      </div>
    </div>
  );
}
