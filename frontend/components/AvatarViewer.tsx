"use client";

"use client";

import dynamic from "next/dynamic";
import type { RefObject } from "react";
import type { PoseLandmarks, HandLandmarks } from "@/hooks/usePoseLandmarker";

const AvatarCanvas = dynamic(() => import("./AvatarCanvas"), { ssr: false });

type Props = {
  glbUrl: string | null;
  loading: boolean;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  leftHandRef: RefObject<HandLandmarks | null>;
  rightHandRef: RefObject<HandLandmarks | null>;
  faceBlendshapesRef: RefObject<number[] | null>;
  mirrorRef: RefObject<boolean>;
  active: boolean;
};

export default function AvatarViewer({ glbUrl, loading, landmarksRef, normLandmarksRef, leftHandRef, rightHandRef, faceBlendshapesRef, mirrorRef, active }: Props) {
  return (
    <div className="w-full h-full rounded-2xl overflow-hidden border border-black/[0.06]">
      <AvatarCanvas glbUrl={glbUrl} loading={loading} landmarksRef={landmarksRef} normLandmarksRef={normLandmarksRef} leftHandRef={leftHandRef} rightHandRef={rightHandRef} faceBlendshapesRef={faceBlendshapesRef} mirrorRef={mirrorRef} webcamActive={active} />
    </div>
  );
}
