"use client";

import { Suspense, useMemo, useRef, useEffect } from "react";
import type { RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
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
  webcamActive: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
};

const BODY_MATERIAL = new THREE.MeshStandardMaterial({
  color: "#a78bfa",
  roughness: 0.65,
  metalness: 0.05,
});

// Camera tuning
const CAM_BASE_Z       = 3.0;
const CAM_MIN_Z        = 0.6;   // close enough to show just face
const CAM_MAX_Z        = 6.0;
const CAM_SHOULDER_REF = 0.25;
const CAM_PAN_SCALE    = 2.5;
const CAM_LERP         = 0.07;
const VIS_THRESHOLD    = 0.3;

// lookAt Y interpolated from face level (close) to body center (far)
const CAM_LOOKAT_Y_CLOSE = 0.8;
const CAM_LOOKAT_Y_FAR   = 0.0;

// ── Video background plane ────────────────────────────────────────────────────
// World-space plane behind avatar. Large enough that camera pan/zoom never
// reveals its edges. VideoTexture maps the webcam feed onto it.

function VideoBackground({
  videoRef, mirrorRef, webcamActive,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  mirrorRef: RefObject<boolean>;
  webcamActive: boolean;
}) {
  const matRef     = useRef<THREE.MeshBasicMaterial>(null);
  const textureRef = useRef<THREE.VideoTexture | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    const mat   = matRef.current;
    if (!webcamActive || !video || !mat) return;

    const tex = new THREE.VideoTexture(video);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter  = THREE.LinearFilter;
    textureRef.current = tex;
    mat.map = tex;
    mat.needsUpdate = true;

    return () => {
      tex.dispose();
      textureRef.current = null;
      if (matRef.current) {
        matRef.current.map = null;
        matRef.current.needsUpdate = true;
      }
    };
  }, [webcamActive, videoRef]);

  // Sync mirror flip every frame
  useFrame(() => {
    const tex = textureRef.current;
    if (!tex) return;
    tex.repeat.x = mirrorRef.current ? -1 : 1;
    tex.offset.x = mirrorRef.current ? 1 : 0;
  });

  if (!webcamActive) return null;

  return (
    // z=-3: behind avatar (at z=0). Plane 20×12 covers view at all zoom levels.
    <mesh position={[0, 0.5, -3]} renderOrder={-1}>
      <planeGeometry args={[20, 12]} />
      <meshBasicMaterial ref={matRef} depthWrite={false} />
    </mesh>
  );
}

// ── Avatar ────────────────────────────────────────────────────────────────────

type AvatarProps = {
  url: string;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  mirrorRef: RefObject<boolean>;
  webcamActive: boolean;
};

function Avatar({ url, landmarksRef, normLandmarksRef, mirrorRef, webcamActive }: AvatarProps) {
  const { scene, camera } = useThree();
  const skeletonRef = useRef<THREE.Skeleton | null>(null);

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

  useEffect(() => {
    if (!webcamActive) {
      camera.position.set(0, 0, CAM_BASE_Z);
      camera.lookAt(0, 0.5, 0);
    }
  }, [webcamActive, camera]);

  useFrame(() => {
    const lms     = landmarksRef.current;
    const normLms = normLandmarksRef.current;
    const sk      = skeletonRef.current;
    if (!lms || !sk || lms.length < 33) {
      if (Math.random() < 0.01)
        console.log("[AvatarCanvas] bail: lms=", lms ? lms.length : null, "sk=", !!sk);
      return;
    }
    driveSkeleton(sk, lms, mirrorRef.current);

    if (!webcamActive || !normLms || normLms.length < 33) return;

    const lm11 = normLms[11];
    const lm12 = normLms[12];
    const lm23 = normLms[23];
    const lm24 = normLms[24];

    // Z zoom from shoulder width
    const shoulderVis = Math.min(lm11.visibility ?? 0, lm12.visibility ?? 0);
    if (shoulderVis > VIS_THRESHOLD) {
      const shoulderW = Math.max(0.05, Math.min(0.6, Math.abs(lm11.x - lm12.x)));
      const targetZ   = Math.max(CAM_MIN_Z, Math.min(CAM_MAX_Z, CAM_SHOULDER_REF * CAM_BASE_Z / shoulderW));
      camera.position.z += (targetZ - camera.position.z) * CAM_LERP;
    }

    // X pan from hip midpoint
    const hipVis = Math.min(lm23.visibility ?? 0, lm24.visibility ?? 0);
    if (hipVis > VIS_THRESHOLD) {
      const hipNormX = (lm23.x + lm24.x) * 0.5;
      const flip     = mirrorRef.current ? 1 : -1;
      const targetX  = (hipNormX - 0.5) * CAM_PAN_SCALE * flip;
      camera.position.x += (targetX - camera.position.x) * CAM_LERP;
    }

    // lookAt Y: face level when close, body center when far
    const t        = Math.max(0, Math.min(1, (camera.position.z - CAM_MIN_Z) / (CAM_MAX_Z - CAM_MIN_Z)));
    const lookAtY  = CAM_LOOKAT_Y_CLOSE + t * (CAM_LOOKAT_Y_FAR - CAM_LOOKAT_Y_CLOSE);
    camera.lookAt(0, lookAtY, 0);
  });

  return (
    <Center>
      <primitive object={scene} />
    </Center>
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

export default function AvatarCanvas({
  glbUrl, loading, landmarksRef, normLandmarksRef,
  mirrorRef, webcamActive, videoRef,
}: Props) {
  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 0, CAM_BASE_Z], fov: 50 }}
        style={{ background: "#09090b" }}
        shadows
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[2, 4, 2]} intensity={1.2} castShadow />
        <directionalLight position={[-2, 2, -1]} intensity={0.4} />

        <VideoBackground videoRef={videoRef} mirrorRef={mirrorRef} webcamActive={webcamActive} />

        {glbUrl ? (
          <Suspense fallback={null}>
            <Avatar
              key={glbUrl}
              url={glbUrl}
              landmarksRef={landmarksRef}
              normLandmarksRef={normLandmarksRef}
              mirrorRef={mirrorRef}
              webcamActive={webcamActive}
            />
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
        <OrbitControls makeDefault enabled={!webcamActive} />
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
