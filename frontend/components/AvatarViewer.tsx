"use client";

"use client";

import dynamic from "next/dynamic";
import type { RefObject } from "react";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

const AvatarCanvas = dynamic(() => import("./AvatarCanvas"), { ssr: false });

type Props = {
  glbUrl: string | null;
  loading: boolean;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  mirrorRef: RefObject<boolean>;
};

export default function AvatarViewer({ glbUrl, loading, landmarksRef, normLandmarksRef, mirrorRef }: Props) {
  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0">
      <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider shrink-0">
        3D Avatar
      </h2>
      <div className="flex-1 min-h-0 rounded-xl overflow-hidden border border-zinc-800">
        <AvatarCanvas glbUrl={glbUrl} loading={loading} landmarksRef={landmarksRef} normLandmarksRef={normLandmarksRef} mirrorRef={mirrorRef} />
      </div>
    </div>
  );
}
