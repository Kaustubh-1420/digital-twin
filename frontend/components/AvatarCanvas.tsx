"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";

type Props = {
  glbUrl: string | null;
  loading: boolean;
};

function PlaceholderFigure() {
  return (
    <group>
      {/* torso */}
      <mesh position={[0, 0.9, 0]}>
        <boxGeometry args={[0.5, 0.7, 0.25]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      {/* head */}
      <mesh position={[0, 1.5, 0]}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#6366f1" />
      </mesh>
      {/* left arm */}
      <mesh position={[-0.38, 0.85, 0]} rotation={[0, 0, 0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      {/* right arm */}
      <mesh position={[0.38, 0.85, 0]} rotation={[0, 0, -0.2]}>
        <boxGeometry args={[0.12, 0.6, 0.12]} />
        <meshStandardMaterial color="#4f46e5" />
      </mesh>
      {/* left leg */}
      <mesh position={[-0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#3730a3" />
      </mesh>
      {/* right leg */}
      <mesh position={[0.15, 0.25, 0]}>
        <boxGeometry args={[0.18, 0.6, 0.18]} />
        <meshStandardMaterial color="#3730a3" />
      </mesh>
    </group>
  );
}

export default function AvatarCanvas({ glbUrl, loading }: Props) {
  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 1, 2.5], fov: 50 }}
        style={{ background: "#09090b" }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[2, 4, 2]} intensity={1.2} />
        <PlaceholderFigure />
        <Grid
          position={[0, -0.02, 0]}
          args={[6, 6]}
          cellSize={0.5}
          cellColor="#27272a"
          sectionColor="#3f3f46"
          fadeDistance={6}
          infiniteGrid
        />
        <OrbitControls target={[0, 0.9, 0]} />
      </Canvas>

      {/* overlays */}
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
