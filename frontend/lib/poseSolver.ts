/**
 * Drives an SMPL-X/VRM skeleton from MediaPipe world landmarks each frame.
 *
 * Approach: swing decomposition — for each bone, compute the quaternion
 * that rotates the T-pose rest direction to the observed direction, then
 * convert world→local by removing the parent's accumulated world rotation.
 *
 * Bones are processed parent-first so world matrices stay consistent.
 */

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

// ── Rest directions in SMPL-X T-pose world space (measured, y-up, person faces +z) ──
// These are the actual directions each bone points when all joint rotations = identity.
// Using these exact values means the avatar is in T-pose when the person stands in T-pose.
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

// ── Bone config: which landmark pair defines each bone's direction ─────────────
// Ordered parent-first so each bone's parent world matrix is current when we process it.
type BoneCfg = {
  name: string;
  from: number; // parent landmark index
  to:   number; // child landmark index
};

const BONE_CFGS: BoneCfg[] = [
  // Spine chain — use shoulder midpoint vs hip midpoint for each segment
  // (approximation: all spine segments share the same observed direction)
  { name: "Spine",         from: MP.LEFT_HIP,      to: MP.LEFT_SHOULDER  },  // replaced with midpoint below
  { name: "Chest",         from: MP.LEFT_HIP,      to: MP.LEFT_SHOULDER  },
  { name: "UpperChest",    from: MP.LEFT_HIP,      to: MP.LEFT_SHOULDER  },

  { name: "Neck",          from: MP.LEFT_SHOULDER, to: MP.LEFT_EAR       },
  { name: "Head",          from: MP.LEFT_EAR,      to: MP.NOSE           },

  // Legs
  { name: "LeftUpperLeg",  from: MP.LEFT_HIP,      to: MP.LEFT_KNEE      },
  { name: "LeftLowerLeg",  from: MP.LEFT_KNEE,     to: MP.LEFT_ANKLE     },
  { name: "RightUpperLeg", from: MP.RIGHT_HIP,     to: MP.RIGHT_KNEE     },
  { name: "RightLowerLeg", from: MP.RIGHT_KNEE,    to: MP.RIGHT_ANKLE    },

  // Arms
  { name: "LeftUpperArm",  from: MP.LEFT_SHOULDER, to: MP.LEFT_ELBOW     },
  { name: "LeftLowerArm",  from: MP.LEFT_ELBOW,    to: MP.LEFT_WRIST     },
  { name: "RightUpperArm", from: MP.RIGHT_SHOULDER,to: MP.RIGHT_ELBOW    },
  { name: "RightLowerArm", from: MP.RIGHT_ELBOW,   to: MP.RIGHT_WRIST    },
];

// Bones that use the spine midpoint direction (hip midpoint → shoulder midpoint)
const SPINE_BONES = new Set(["Spine", "Chest", "UpperChest"]);
// Bones that use the neck midpoint (shoulder midpoint → ear midpoint)
const NECK_BONES = new Set(["Neck"]);

// ── Helpers ───────────────────────────────────────────────────────────────────

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _q1 = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();

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

const _parentWorldQ = new THREE.Quaternion();
const _worldQ       = new THREE.Quaternion();
const _localQ       = new THREE.Quaternion();

export function driveSkeleton(skeleton: THREE.Skeleton, lms: PoseLandmarks): void {
  // Build bone name → THREE.Bone map
  const bones: Map<string, THREE.Bone> = new Map();
  skeleton.bones.forEach((b) => bones.set(b.name, b));

  // Precompute midpoints used by spine/neck
  const hipMid      = midpoint(lms, MP.LEFT_HIP,      MP.RIGHT_HIP);
  const shoulderMid = midpoint(lms, MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER);
  const earMid      = midpoint(lms, MP.LEFT_EAR,       MP.RIGHT_EAR);
  const spineDir    = midDir(hipMid, shoulderMid);
  const neckDir     = midDir(shoulderMid, earMid);

  for (const cfg of BONE_CFGS) {
    const bone = bones.get(cfg.name);
    const restDir = REST[cfg.name];
    if (!bone || !restDir) continue;

    // 1. Observed direction (world space)
    let observed: THREE.Vector3;
    if (SPINE_BONES.has(cfg.name)) {
      observed = spineDir;
    } else if (NECK_BONES.has(cfg.name)) {
      observed = neckDir;
    } else {
      observed = lmDir(lms, cfg.from, cfg.to);
    }

    // 2. World rotation: maps T-pose rest direction → observed direction
    _worldQ.setFromUnitVectors(restDir, observed);

    // 3. Parent world quaternion (requires parent world matrix to be current)
    _parentWorldQ.identity();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(_parentWorldQ);
    }

    // 4. Local rotation = parentWorldQ⁻¹ × worldQ
    _q1.copy(_parentWorldQ).invert();
    _localQ.multiplyQuaternions(_q1, _worldQ);

    // 5. SLERP toward target (smooths residual jitter after landmark filtering)
    bone.quaternion.slerp(_localQ, 0.35);
    bone.updateWorldMatrix(false, false);
  }
}
