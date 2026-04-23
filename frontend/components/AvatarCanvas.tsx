"use client";

import { Suspense, useMemo } from "react";
import type { RefObject } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF, OrbitControls, Grid, Center } from "@react-three/drei";
import * as THREE from "three";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

type Props = {
  glbUrl: string | null;
  loading: boolean;
  landmarksRef: RefObject<PoseLandmarks | null>;
};

const BODY_MATERIAL = new THREE.MeshStandardMaterial({
  color: "#a78bfa",
  roughness: 0.65,
  metalness: 0.05,
});

function Avatar({ url }: { url: string }) {
  const { scene } = useGLTF(url);

  useMemo(() => {
    scene.traverse((obj) => {
      if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
        (obj as THREE.SkinnedMesh).material = BODY_MATERIAL;
        (obj as THREE.SkinnedMesh).castShadow = true;
      }
    });
  }, [scene]);

  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

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

export default function AvatarCanvas({ glbUrl, loading, landmarksRef }: Props) {
  // landmarksRef is read each frame inside useFrame (task 16) — no prop drilling into Canvas needed
  void landmarksRef;

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
            <Avatar key={glbUrl} url={glbUrl} />
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
