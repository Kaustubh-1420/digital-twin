"use client";

import { Suspense, useMemo, useRef } from "react";
import type { RefObject } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, OrbitControls, Grid, Center } from "@react-three/drei";
import * as THREE from "three";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";
import { driveSkeleton } from "@/lib/poseSolver";

type Props = {
  glbUrl: string | null;
  loading: boolean;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  mirrorRef: RefObject<boolean>;
};

const BODY_MATERIAL = new THREE.MeshStandardMaterial({
  color: "#a78bfa",
  roughness: 0.65,
  metalness: 0.05,
});

// ── Avatar (with skeleton driving) ───────────────────────────────────────────

// X scale: how many Three.js units = full screen width (camera fov=50 at z=3)
const ROOT_X_SCALE = 4.0;
// Z scale: world landmark z is in metric (~±0.3 m range), map to scene units
const ROOT_Z_SCALE = 2.5;
// Lerp factor for root position smoothing (lower = smoother but laggier)
const ROOT_LERP   = 0.12;
// Hip visibility threshold for root translation
const HIP_VIS_THRESHOLD = 0.45;

type AvatarProps = {
  url: string;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  mirrorRef: RefObject<boolean>;
};

function Avatar({ url, landmarksRef, normLandmarksRef, mirrorRef }: AvatarProps) {
  const { scene } = useGLTF(url);
  const skeletonRef  = useRef<THREE.Skeleton | null>(null);
  const rootGroupRef = useRef<THREE.Group>(null);

  useMemo(() => {
    scene.traverse((obj) => {
      if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
        const sm = obj as THREE.SkinnedMesh;
        sm.material = BODY_MATERIAL;
        sm.castShadow = true;
        skeletonRef.current = sm.skeleton;
      }
    });
  }, [scene]);

  useFrame(() => {
    const lms     = landmarksRef.current;
    const normLms = normLandmarksRef.current;
    const sk      = skeletonRef.current;
    const root    = rootGroupRef.current;
    if (!lms || !sk || lms.length < 33) {
      if (Math.random() < 0.01) {
        console.log("[AvatarCanvas] useFrame bail: lms=", lms ? lms.length : null, "sk=", !!sk);
      }
      return;
    }
    driveSkeleton(sk, lms, mirrorRef.current);

    // Root translation: drive avatar position from hip midpoint
    if (root && normLms && normLms.length >= 33) {
      const lm23 = normLms[23]; // left hip (screen-space, 0-1)
      const lm24 = normLms[24]; // right hip
      if ((lm23.visibility ?? 0) > HIP_VIS_THRESHOLD && (lm24.visibility ?? 0) > HIP_VIS_THRESHOLD) {
        const hipNormX = (lm23.x + lm24.x) * 0.5;
        // Screen x=0 is camera-left; front-facing camera → flip for natural mirroring
        const flip = mirrorRef.current ? 1 : -1;
        const targetX = (hipNormX - 0.5) * ROOT_X_SCALE * flip;
        // World landmark z: negative = closer to camera → positive scene Z
        const hipWorldZ = (lms[23].z + lms[24].z) * 0.5;
        const targetZ = -hipWorldZ * ROOT_Z_SCALE;
        root.position.x += (targetX - root.position.x) * ROOT_LERP;
        root.position.z += (targetZ - root.position.z) * ROOT_LERP;
      }
    }
  });

  return (
    <group ref={rootGroupRef}>
      <Center>
        <primitive object={scene} />
      </Center>
    </group>
  );
}

// ── Placeholder ───────────────────────────────────────────────────────────────

function PlaceholderFigure() {
  return (
    <group>
      <mesh position={[0, 0.9, 0]}>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      <mesh position={[0, 1.5, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#6366f1" />
      </mesh>
      <mesh position={[-0.38, 0.85, 0]} rotation={[0, 0, 0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      <mesh position={[0.38, 0.85, 0]} rotation={[0, 0, -0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      <mesh position={[-0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#3730a3" />
      </mesh>
      <mesh position={[0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#3730a3" />
      </mesh>
    </group>
  );
}

// ── Canvas ────────────────────────────────────────────────────────────────────

export default function AvatarCanvas({ glbUrl, loading, landmarksRef, normLandmarksRef, mirrorRef }: Props) {
  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 3], fov: 50 }}
        style={{ background: "#09090b" }}
        shadows
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[2, 4, 2]} intensity={1.2} castShadow />
        <directionalLight position={[-2, 2, -1]} intensity={0.4} />

        {glbUrl ? (
          <Suspense fallback={null}>
            <Avatar key={glbUrl} url={glbUrl} landmarksRef={landmarksRef} normLandmarksRef={normLandmarksRef} mirrorRef={mirrorRef} />
          </Suspense>
        ) : (
          <Center>
            <PlaceholderFigure />
          </Center>
        )}

        <Grid
          position={[0, -0.9, 0]}
          args={[8, 8]}
          cellSize={0.5}
          cellColor="#27272a"
          sectionColor="#3f3f46"
          fadeDistance={7}
          infiniteGrid
        />
        <OrbitControls makeDefault />
      </Canvas>

      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/70 gap-3">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-zinc-300">Generating your avatar…</span>
        </div>
      )}
      {!glbUrl && !loading && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2">
          <span className="text-xs text-zinc-600 bg-zinc-950/80 px-3 py-1 rounded-full">
            Your avatar will appear here
          </span>
        </div>
      )}
    </div>
  );
}
