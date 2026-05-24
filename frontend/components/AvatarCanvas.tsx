"use client";

import { Suspense, useMemo, useRef, useEffect } from "react";
import type { RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls, ContactShadows, Center } from "@react-three/drei";
import * as THREE from "three";
import type { PoseLandmarks, HandLandmarks } from "@/hooks/usePoseLandmarker";
import { driveSkeleton, driveHands, driveJawEyes } from "@/lib/poseSolver";
import { driveShowcase } from "@/lib/showcaseLoop";
import { EXPR_W } from "@/lib/exprMapping";

const FACE_ENABLED = true;
const SAMPLE_GLB_URL = "/sample-avatar.glb";

type Props = {
  glbUrl: string | null;
  loading: boolean;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  leftHandRef: RefObject<HandLandmarks | null>;
  rightHandRef: RefObject<HandLandmarks | null>;
  faceBlendshapesRef: RefObject<number[] | null>;
  mirrorRef: RefObject<boolean>;
  webcamActive: boolean;
};

const BODY_MATERIAL = new THREE.MeshStandardMaterial({
  color: "#c8b4a0",
  roughness: 0.65,
  metalness: 0.05,
});

// Camera tuning
const CAM_BASE_Z         = 3.0;
const CAM_MIN_Z          = 0.3;
const CAM_MAX_Z          = 6.0;
const CAM_SHOULDER_REF   = 0.10;
const CAM_PAN_SCALE      = 2.5;
const CAM_LERP           = 0.07;
const VIS_THRESHOLD      = 0.3;
// Camera Y rises to face height when zoomed in, returns to center when far
const CAM_Y_CLOSE        = 0.6;
const CAM_Y_FAR          = 0.0;
const CAM_LOOKAT_Y       = 0.75; // fixed — roughly neck/head level

// Blendshape indices driven by jaw bone — skip these in the expression PCA matrix
// to prevent cross-talk (jaw open triggering wild mouth expression deformations)
// Keep: brow (1-8), eyeBlink (9-10), eyeLook (11-18), eyeSquint/Wide (19-22), noseSneer (50-51)
// eyeBlink stays in matrix — eyelid closure is expression PCA, not eye bone rotation
const BONE_DRIVEN_BS = new Set([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);

const NEUTRAL_CALIB_FRAMES = 30;
const _neutralAccum: number[] = new Array(52).fill(0);
let _neutralCount = 0;
let _neutralBaseline: number[] | null = null;

// Returns calibrated scores after baseline subtraction, or null during warmup
function driveMorphTargets(mesh: THREE.SkinnedMesh, scores: number[]): number[] | null {
  const dict = mesh.morphTargetDictionary;
  const inf  = mesh.morphTargetInfluences;
  if (!dict || !inf) return null;

  // Collect first N frames to build neutral baseline, skip driving until ready
  if (_neutralCount < NEUTRAL_CALIB_FRAMES) {
    for (let j = 0; j < 52; j++) _neutralAccum[j] += scores[j];
    _neutralCount++;
    if (_neutralCount === NEUTRAL_CALIB_FRAMES) {
      _neutralBaseline = _neutralAccum.map(v => v / NEUTRAL_CALIB_FRAMES);
    }
    return null;
  }

  // Subtract user's resting face from raw scores; clamp to [0,1]
  const calibrated = scores.map((s, j) => Math.max(0, Math.min(1, s - _neutralBaseline![j])));

  // calibrated (52,) @ EXPR_W (52×100) → expr params (100,)
  // Bone-driven indices excluded to prevent jaw/mouth cross-talk
  for (let i = 0; i < 100; i++) {
    let v = 0;
    for (let j = 0; j < 52; j++) {
      if (BONE_DRIVEN_BS.has(j)) continue;
      v += calibrated[j] * EXPR_W[j][i];
    }
    const slotIdx = dict[`expr_${i}`];
    if (slotIdx !== undefined) {
      inf[slotIdx] += (v - inf[slotIdx]) * 0.15;
    }
  }

  return calibrated;
}


// ── Avatar ────────────────────────────────────────────────────────────────────

type AvatarProps = {
  url: string;
  landmarksRef: RefObject<PoseLandmarks | null>;
  normLandmarksRef: RefObject<PoseLandmarks | null>;
  leftHandRef: RefObject<HandLandmarks | null>;
  rightHandRef: RefObject<HandLandmarks | null>;
  faceBlendshapesRef: RefObject<number[] | null>;
  mirrorRef: RefObject<boolean>;
  webcamActive: boolean;
  showcaseMode?: boolean;
};

function Avatar({ url, landmarksRef, normLandmarksRef, leftHandRef, rightHandRef, faceBlendshapesRef, mirrorRef, webcamActive, showcaseMode = false }: AvatarProps) {
  const { scene: gltfScene } = useGLTF(url);
  const { camera } = useThree();
  const skeletonRef = useRef<THREE.Skeleton | null>(null);
  const meshRef     = useRef<THREE.SkinnedMesh | null>(null);

  useMemo(() => {
    gltfScene.traverse((obj) => {
      if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
        const sm = obj as THREE.SkinnedMesh;
        sm.material = BODY_MATERIAL;
        sm.castShadow = true;
        skeletonRef.current = sm.skeleton;
        meshRef.current     = sm;

      }
    });
  }, [gltfScene]);

  useEffect(() => {
    if (!webcamActive) {
      camera.position.set(0, 0, CAM_BASE_Z);
      camera.lookAt(0, CAM_LOOKAT_Y, 0);
    }
  }, [webcamActive, camera]);

  useFrame((state) => {
    const sk = skeletonRef.current;
    if (!sk) return;

    if (showcaseMode) {
      driveShowcase(sk, state.clock.elapsedTime);
      return;
    }

    const lms     = landmarksRef.current;
    const normLms = normLandmarksRef.current;
    if (!lms || lms.length < 33) return;
    const isMirror = mirrorRef.current;
    driveSkeleton(sk, lms, isMirror);
    driveHands(sk,
      isMirror ? rightHandRef.current : leftHandRef.current,
      isMirror ? leftHandRef.current  : rightHandRef.current,
    );

    if (FACE_ENABLED && meshRef.current && faceBlendshapesRef.current) {
      const calibrated = driveMorphTargets(meshRef.current, faceBlendshapesRef.current);
      if (calibrated) driveJawEyes(sk, calibrated);
    }

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

    // Raise camera Y when close (face height) → head-on view, not worm's-eye
    const t       = Math.max(0, Math.min(1, (camera.position.z - CAM_MIN_Z) / (CAM_MAX_Z - CAM_MIN_Z)));
    const targetY = CAM_Y_CLOSE + t * (CAM_Y_FAR - CAM_Y_CLOSE);
    camera.position.y += (targetY - camera.position.y) * CAM_LERP;
    camera.lookAt(0, CAM_LOOKAT_Y, 0);
  });

  if (showcaseMode) {
    return (
      <Center position={[0, -0.25, 0]}>
        <primitive object={gltfScene} />
      </Center>
    );
  }
  return (
    <Center>
      <primitive object={gltfScene} />
    </Center>
  );
}

// ── Placeholder ───────────────────────────────────────────────────────────────

function PlaceholderFigure() {
  return (
    <group>
      <mesh position={[0, 0.9, 0]}>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color="#aaaaaa" />
      </mesh>
      <mesh position={[0, 1.5, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#bbbbbb" />
      </mesh>
      <mesh position={[-0.38, 0.85, 0]} rotation={[0, 0, 0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#aaaaaa" />
      </mesh>
      <mesh position={[0.38, 0.85, 0]} rotation={[0, 0, -0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#aaaaaa" />
      </mesh>
      <mesh position={[-0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#999999" />
      </mesh>
      <mesh position={[0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#999999" />
      </mesh>
    </group>
  );
}

// ── Canvas ────────────────────────────────────────────────────────────────────

export default function AvatarCanvas({
  glbUrl, loading, landmarksRef, normLandmarksRef, leftHandRef, rightHandRef, faceBlendshapesRef, mirrorRef, webcamActive,
}: Props) {
  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 0, CAM_BASE_Z], fov: 50 }}
        gl={{ alpha: true }}
        style={{ background: "transparent" }}
        shadows
      >
        <ambientLight intensity={0.5} />
        {/* Key light — front-left, warm white */}
        <directionalLight position={[3, 5, 3]} intensity={1.8} color="#fff8f0" castShadow
          shadow-mapSize={[1024, 1024]} shadow-camera-near={0.5} shadow-camera-far={20} />
        {/* Fill light — front-right, cool white */}
        <directionalLight position={[-3, 3, 2]} intensity={0.8} color="#f0f4ff" />
        {/* Rim light — behind, separates avatar from background */}
        <directionalLight position={[0, 4, -4]} intensity={1.0} color="#ffffff" />

        <ContactShadows position={[0, -0.91, 0]} opacity={0.25} scale={4} blur={2.5} far={1.5} />

        {!loading && (glbUrl ? (
          <Suspense fallback={null}>
            <Avatar
              key={glbUrl}
              url={glbUrl}
              landmarksRef={landmarksRef}
              normLandmarksRef={normLandmarksRef}
              leftHandRef={leftHandRef}
              rightHandRef={rightHandRef}
              faceBlendshapesRef={faceBlendshapesRef}
              mirrorRef={mirrorRef}
              webcamActive={webcamActive}
            />
          </Suspense>
        ) : (
          <Suspense fallback={null}>
            <Avatar
              key="showcase"
              url={SAMPLE_GLB_URL}
              landmarksRef={landmarksRef}
              normLandmarksRef={normLandmarksRef}
              leftHandRef={leftHandRef}
              rightHandRef={rightHandRef}
              faceBlendshapesRef={faceBlendshapesRef}
              mirrorRef={mirrorRef}
              webcamActive={false}
              showcaseMode
            />
          </Suspense>
        ))}

        <OrbitControls makeDefault enabled={!webcamActive} />
      </Canvas>

      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#fafaf8]/80 gap-3">
          <div className="w-8 h-8 border-2 border-[#b07a5e] border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-black/50">Generating your avatar…</span>
        </div>
      )}
    </div>
  );
}
