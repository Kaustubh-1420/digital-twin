import * as THREE from "three";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

// Toggle to see per-frame diagnostics in the browser console.
const DEBUG_POSE = true;
let _dbgFrame = 0;

// Confirm this module version is loaded — remove when solver is stable
console.log("[PoseSolver] module loaded — local-space swing decomp, y/z flip ON, threshold=0.45");

// ── MediaPipe landmark indices ────────────────────────────────────────────────
const MP = {
  NOSE: 0,
  LEFT_EAR: 7,  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,    RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,    RIGHT_WRIST: 16,
  LEFT_HIP: 23,      RIGHT_HIP: 24,
  LEFT_KNEE: 25,     RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,    RIGHT_ANKLE: 28,
  LEFT_FOOT: 31,     RIGHT_FOOT: 32,
} as const;

const VISIBILITY_THRESHOLD = 0.45;

// ── Rest directions ───────────────────────────────────────────────────────────
// MediaPipe world landmarks are Y-UP (hip-centred, metric). No coordinate
// transform needed — compare restDir to observed direction directly.
// Using clean cardinal directions to avoid compounding small Z errors.
const R = (x: number, y: number, z: number) =>
  new THREE.Vector3(x, y, z).normalize();

const REST: Record<string, THREE.Vector3> = {
  Spine:         R( 0,     1,     0),
  LeftUpperArm:  R( 0.99, -0.13,  0),
  RightUpperArm: R(-0.99, -0.13,  0),
  LeftLowerArm:  R( 1,     0,     0),
  RightLowerArm: R(-1,     0,     0),
  LeftUpperLeg:  R( 0,    -1,     0),
  LeftLowerLeg:  R( 0,    -1,     0),
  RightUpperLeg: R( 0,    -1,     0),
  RightLowerLeg: R( 0,    -1,     0),
};

// ── Bone configs ──────────────────────────────────────────────────────────────
type BoneCfg = {
  name: string;
  from: number;
  to:   number;
  vis?: number[];
};

// Normal mode: anatomical — your left arm drives avatar's left arm.
const BONE_CFGS: BoneCfg[] = [
  { name: "Spine",      from: MP.LEFT_HIP, to: MP.LEFT_SHOULDER, vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER] },

  { name: "LeftUpperLeg",  from: MP.LEFT_HIP,    to: MP.LEFT_KNEE   },
  { name: "LeftLowerLeg",  from: MP.LEFT_KNEE,   to: MP.LEFT_ANKLE  },
  { name: "RightUpperLeg", from: MP.RIGHT_HIP,   to: MP.RIGHT_KNEE  },
  { name: "RightLowerLeg", from: MP.RIGHT_KNEE,  to: MP.RIGHT_ANKLE },

  { name: "LeftUpperArm",  from: MP.LEFT_SHOULDER,  to: MP.LEFT_ELBOW  },
  { name: "LeftLowerArm",  from: MP.LEFT_ELBOW,     to: MP.LEFT_WRIST  },
  { name: "RightUpperArm", from: MP.RIGHT_SHOULDER, to: MP.RIGHT_ELBOW },
  { name: "RightLowerArm", from: MP.RIGHT_ELBOW,    to: MP.RIGHT_WRIST },
];

// Mirror mode: selfie/webcam — swap left↔right so the avatar looks like a reflection.
const BONE_CFGS_MIRROR: BoneCfg[] = [
  { name: "Spine",      from: MP.LEFT_HIP, to: MP.LEFT_SHOULDER, vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER] },

  // Legs swapped
  { name: "LeftUpperLeg",  from: MP.RIGHT_HIP,   to: MP.RIGHT_KNEE  },
  { name: "LeftLowerLeg",  from: MP.RIGHT_KNEE,  to: MP.RIGHT_ANKLE },
  { name: "RightUpperLeg", from: MP.LEFT_HIP,    to: MP.LEFT_KNEE   },
  { name: "RightLowerLeg", from: MP.LEFT_KNEE,   to: MP.LEFT_ANKLE  },

  // Arms swapped
  { name: "LeftUpperArm",  from: MP.RIGHT_SHOULDER, to: MP.RIGHT_ELBOW, vis: [MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW] },
  { name: "LeftLowerArm",  from: MP.RIGHT_ELBOW,    to: MP.RIGHT_WRIST, vis: [MP.RIGHT_ELBOW, MP.RIGHT_WRIST]   },
  { name: "RightUpperArm", from: MP.LEFT_SHOULDER,  to: MP.LEFT_ELBOW,  vis: [MP.LEFT_SHOULDER, MP.LEFT_ELBOW]  },
  { name: "RightLowerArm", from: MP.LEFT_ELBOW,     to: MP.LEFT_WRIST,  vis: [MP.LEFT_ELBOW, MP.LEFT_WRIST]    },
];

const SPINE_BONES = new Set(["Spine"]);

// ── Module-level state ────────────────────────────────────────────────────────

const _prevBoneQ = new Map<string, THREE.Quaternion>();

// ── Helpers ───────────────────────────────────────────────────────────────────

const _v1 = new THREE.Vector3();   // scratch: observed in parent-local space
const _q1 = new THREE.Quaternion(); // scratch: parentWorldQ.inverse
const _localQ = new THREE.Quaternion();
const _parentWorldQ = new THREE.Quaternion();

function landmarkOk(lms: PoseLandmarks, idx: number): boolean {
  const vis = lms[idx]?.visibility;
  return vis === undefined || vis >= VISIBILITY_THRESHOLD;
}

function allVisible(lms: PoseLandmarks, indices: number[]): boolean {
  return indices.every((i) => landmarkOk(lms, i));
}

function midpoint(lms: PoseLandmarks, a: number, b: number): THREE.Vector3 {
  return new THREE.Vector3(
    (lms[a].x + lms[b].x) / 2,
    (lms[a].y + lms[b].y) / 2,
    (lms[a].z + lms[b].z) / 2,
  );
}

function lmDir(lms: PoseLandmarks, from: number, to: number): THREE.Vector3 {
  _v1.set(lms[to].x - lms[from].x, lms[to].y - lms[from].y, lms[to].z - lms[from].z);
  return _v1.clone().normalize();
}


function fv(v: THREE.Vector3): string {
  return `(${v.x.toFixed(3)}, ${v.y.toFixed(3)}, ${v.z.toFixed(3)})`;
}

function fq(q: THREE.Quaternion): string {
  const e = new THREE.Euler().setFromQuaternion(q, "XYZ");
  const deg = (r: number) => ((r * 180) / Math.PI).toFixed(1);
  return `euler(${deg(e.x)}°,${deg(e.y)}°,${deg(e.z)}°)`;
}

// ── Main export ───────────────────────────────────────────────────────────────

export function driveSkeleton(
  skeleton: THREE.Skeleton,
  lms: PoseLandmarks,
  mirror = false,
): void {
  // MediaPipe world landmarks are Y-DOWN (camera/OpenCV convention).
  // Negate Y and Z to convert to Y-UP OpenGL space matching our REST directions.
  const src: PoseLandmarks = lms.map(lm => ({
    x: lm.x, y: -lm.y, z: -lm.z, visibility: lm.visibility,
  }));

  _dbgFrame++;
  const logFrame = DEBUG_POSE && _dbgFrame % 30 === 1; // log every 30th frame
  const logFirst = DEBUG_POSE && _dbgFrame <= 3;       // log first 3 frames always

  const cfgs = mirror ? BONE_CFGS_MIRROR : BONE_CFGS;

  const bones: Map<string, THREE.Bone> = new Map();
  skeleton.bones.forEach((b) => bones.set(b.name, b));

  const shoulderMid = midpoint(src, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER);
  // Hip midpoint ≈ world origin by MediaPipe convention — valid even when hips out of frame.
  // spineDir = direction from origin to shoulderMid.
  const spineDir = shoulderMid.clone().normalize();

  if (logFrame || logFirst) {
    console.group(`[PoseSolver] frame=${_dbgFrame} mirror=${mirror}`);
    console.log("  shldL  :", fv(new THREE.Vector3(src[MP.LEFT_SHOULDER].x,  src[MP.LEFT_SHOULDER].y,  src[MP.LEFT_SHOULDER].z)));
    console.log("  shldR  :", fv(new THREE.Vector3(src[MP.RIGHT_SHOULDER].x, src[MP.RIGHT_SHOULDER].y, src[MP.RIGHT_SHOULDER].z)));
    console.log("  shoulderMid:", fv(shoulderMid), "  ← Y should be POSITIVE");
    console.log("  spineDir   :", fv(spineDir),    "  ← Y should be POSITIVE (~1)");
    console.groupEnd();
  }

  for (const cfg of cfgs) {
    const bone    = bones.get(cfg.name);
    const restDir = REST[cfg.name];
    if (!bone || !restDir) continue;

    const visIndices = cfg.vis ?? [cfg.from, cfg.to];
    if (!allVisible(src, visIndices)) {
      if (logFirst || logFrame) {
        const scores = visIndices.map(i => `lm${i}=${lms[i]?.visibility?.toFixed(2) ?? "?"}`).join(" ");
        console.log(`[PoseSolver] ${cfg.name}: SKIPPED visibility: ${scores} (threshold=${VISIBILITY_THRESHOLD})`);
      }
      continue;
    }

    let observed: THREE.Vector3;
    if (SPINE_BONES.has(cfg.name)) observed = spineDir;
    else                           observed = lmDir(src, cfg.from, cfg.to);

    // Swing-decompose in parent-local space.
    // Computing worldQ first (setFromUnitVectors against static restDir) is WRONG
    // when a parent bone is driven — the arm's T-pose world direction changes with
    // the parent rotation, so restDir is no longer the right reference in world space.
    // The correct formulation: rotate observed into parent-local space, then
    // setFromUnitVectors(restDir, localObs) gives localQ directly.
    _parentWorldQ.identity();
    if (bone.parent) bone.parent.getWorldQuaternion(_parentWorldQ);
    _q1.copy(_parentWorldQ).invert();          // q1 = parentWorldQ.inverse
    _v1.copy(observed).applyQuaternion(_q1);   // v1 = observed in parent-local space
    _localQ.setFromUnitVectors(restDir, _v1);

    const prev = _prevBoneQ.get(cfg.name);
    const dot  = prev ? Math.abs(prev.dot(_localQ)) : 1;

    if (logFrame || logFirst) {
      console.log(
        `[PoseSolver] ${cfg.name.padEnd(14)}`,
        `rest=${fv(restDir)}`,
        `obsWorld=${fv(observed)}`,
        `obsLocal=${fv(_v1)}`,
        `dot(rest,obsLocal)=${restDir.dot(_v1).toFixed(3)}`,
        `localQ=${fq(_localQ)}`,
        `dot(prev,cur)=${dot.toFixed(3)}`,
      );
    }

    if (!_prevBoneQ.has(cfg.name)) _prevBoneQ.set(cfg.name, new THREE.Quaternion());
    _prevBoneQ.get(cfg.name)!.copy(_localQ);

    bone.quaternion.slerp(_localQ, 0.35);
    bone.updateWorldMatrix(false, false);
  }
}

export function resetSkeletonDriverState(): void {
  _prevBoneQ.clear();
  _dbgFrame = 0;
}
