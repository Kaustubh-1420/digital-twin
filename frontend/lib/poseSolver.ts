import * as THREE from "three";
import type { PoseLandmarks } from "@/hooks/usePoseLandmarker";

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

const VISIBILITY_THRESHOLD = 0.6;

// Reject bone updates where the quaternion delta exceeds this angle (radians).
// Prevents shoulder-pop singularities from flipping the arm 180° in one frame.
const MAX_DELTA_HALF_ANGLE_COS = Math.cos(1.2 / 2);

// ── Rest directions ───────────────────────────────────────────────────────────
// Measured from SMPL-X T-pose joint positions. MediaPipe world landmarks share
// the same axis convention as SMPL-X: x = person's left (+x), y = up, z = toward camera.
// No coordinate transform needed — just compare restDir to observed direction directly.
const R = (x: number, y: number, z: number) =>
  new THREE.Vector3(x, y, z).normalize();

const REST: Record<string, THREE.Vector3> = {
  Spine:         R(-0.02,  0.97, -0.24),
  Chest:         R( 0.07,  1.00, -0.04),
  UpperChest:    R(-0.19,  0.86,  0.47),
  Neck:          R(-0.07,  0.98, -0.19),
  Head:          R( 0.15,  0.98,  0.13),

  LeftUpperArm:  R( 0.95, -0.27, -0.16),
  LeftLowerArm:  R( 1.00,  0.09, -0.01),
  RightUpperArm: R(-0.99, -0.13, -0.10),
  RightLowerArm: R(-1.00, -0.02, -0.06),

  LeftUpperLeg:  R( 0.14, -0.99, -0.02),
  LeftLowerLeg:  R(-0.11, -0.99, -0.08),
  RightUpperLeg: R(-0.12, -0.99, -0.05),
  RightLowerLeg: R( 0.04, -1.00, -0.05),
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
  // Only drive Spine for the trunk — Chest/UpperChest inherit and stay in bind pose.
  // Driving all three compounds rotations and destroys the mesh.
  { name: "Spine",      from: MP.LEFT_HIP, to: MP.LEFT_SHOULDER, vis: [MP.LEFT_HIP, MP.RIGHT_HIP, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER] },
  { name: "Neck",       from: MP.LEFT_SHOULDER, to: MP.LEFT_EAR,  vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER, MP.LEFT_EAR, MP.RIGHT_EAR] },
  { name: "Head",       from: MP.LEFT_EAR,      to: MP.NOSE,       vis: [MP.LEFT_EAR, MP.RIGHT_EAR, MP.NOSE] },

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
// Your physical right arm drives the avatar's left arm (which appears on the right side
// of the screen), matching the natural selfie-cam experience.
const BONE_CFGS_MIRROR: BoneCfg[] = [
  { name: "Spine",      from: MP.LEFT_HIP, to: MP.LEFT_SHOULDER, vis: [MP.LEFT_HIP, MP.RIGHT_HIP, MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER] },
  { name: "Neck",       from: MP.LEFT_SHOULDER, to: MP.LEFT_EAR,  vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER, MP.LEFT_EAR, MP.RIGHT_EAR] },
  { name: "Head",       from: MP.LEFT_EAR,      to: MP.NOSE,       vis: [MP.LEFT_EAR, MP.RIGHT_EAR, MP.NOSE] },

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
const NECK_BONES  = new Set(["Neck"]);

// ── Module-level state ────────────────────────────────────────────────────────

const _prevBoneQ = new Map<string, THREE.Quaternion>();

// ── Helpers ───────────────────────────────────────────────────────────────────

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q1 = new THREE.Quaternion();
const _localQ = new THREE.Quaternion();
const _parentWorldQ = new THREE.Quaternion();
const _worldQ = new THREE.Quaternion();

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

function midDir(from: THREE.Vector3, to: THREE.Vector3): THREE.Vector3 {
  _v2.subVectors(to, from).normalize();
  return _v2.clone();
}

// ── Main export ───────────────────────────────────────────────────────────────

export function driveSkeleton(
  skeleton: THREE.Skeleton,
  lms: PoseLandmarks,
  mirror = false,
): void {
  // MediaPipe world landmarks: Y-down, Z-into-screen (OpenCV/camera convention).
  // SMPL-X rest directions: Y-up, Z-toward-viewer (OpenGL convention).
  // Negate Y and Z to align the two spaces.
  const src: PoseLandmarks = lms.map(lm => ({
    x: lm.x, y: -lm.y, z: -lm.z, visibility: lm.visibility,
  }));

  const cfgs = mirror ? BONE_CFGS_MIRROR : BONE_CFGS;

  const bones: Map<string, THREE.Bone> = new Map();
  skeleton.bones.forEach((b) => bones.set(b.name, b));

  const hipMid      = midpoint(src, MP.LEFT_HIP,      MP.RIGHT_HIP);
  const shoulderMid = midpoint(src, MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER);
  const earMid      = midpoint(src, MP.LEFT_EAR,       MP.RIGHT_EAR);
  const spineDir    = midDir(hipMid, shoulderMid);
  const neckDir     = midDir(shoulderMid, earMid);

  for (const cfg of cfgs) {
    const bone    = bones.get(cfg.name);
    const restDir = REST[cfg.name];
    if (!bone || !restDir) continue;

    const visIndices = cfg.vis ?? [cfg.from, cfg.to];
    if (!allVisible(src, visIndices)) continue;

    let observed: THREE.Vector3;
    if (SPINE_BONES.has(cfg.name))     observed = spineDir;
    else if (NECK_BONES.has(cfg.name)) observed = neckDir;
    else                               observed = lmDir(src, cfg.from, cfg.to);

    _worldQ.setFromUnitVectors(restDir, observed);

    _parentWorldQ.identity();
    if (bone.parent) bone.parent.getWorldQuaternion(_parentWorldQ);
    _q1.copy(_parentWorldQ).invert();
    _localQ.multiplyQuaternions(_q1, _worldQ);

    // Hysteresis: reject frame if delta > ~69° to catch shoulder-pop singularities
    const prev = _prevBoneQ.get(cfg.name);
    if (prev) {
      if (Math.abs(prev.dot(_localQ)) < MAX_DELTA_HALF_ANGLE_COS) continue;
    }
    if (!_prevBoneQ.has(cfg.name)) _prevBoneQ.set(cfg.name, new THREE.Quaternion());
    _prevBoneQ.get(cfg.name)!.copy(_localQ);

    bone.quaternion.slerp(_localQ, 0.35);
    bone.updateWorldMatrix(false, false);
  }
}

export function resetSkeletonDriverState(): void {
  _prevBoneQ.clear();
}
