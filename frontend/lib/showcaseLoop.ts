import * as THREE from "three";

// Showcase loop — drives the sample avatar with a hand-authored animation
// so the landing page shows a moving figure before the visitor uploads anything.
// Bypasses the MediaPipe pose pipeline; sets bone quaternions directly.

const LOOP_DURATION = 11;
const ARM_REST_ANGLE = 1.15; // ~66°, leaves clearance from torso
const BREATHE_HZ = 0.3;     // breathing cycles per second
const BREATHE_AMP = 0.022;  // ~1.3° spine pitch

const _AXIS_X = new THREE.Vector3(1, 0, 0);
const _AXIS_Y = new THREE.Vector3(0, 1, 0);
const _AXIS_Z = new THREE.Vector3(0, 0, 1);
const _SCRATCH_Q = new THREE.Quaternion();

const _DRIVEN_BONES = [
  "LeftUpperArm", "RightUpperArm",
  "Neck", "Head",
  "Spine", "Chest",
  "Jaw",
];

function bell(u: number): number {
  return Math.sin(u * Math.PI);
}

function setAxisAngle(skeleton: THREE.Skeleton, name: string, axis: THREE.Vector3, angle: number) {
  const b = skeleton.getBoneByName(name);
  if (!b) return;
  b.quaternion.setFromAxisAngle(axis, angle);
}

function resetDriven(skeleton: THREE.Skeleton) {
  for (const n of _DRIVEN_BONES) {
    const b = skeleton.getBoneByName(n);
    if (b) b.quaternion.identity();
  }
}

function applyRestPose(skeleton: THREE.Skeleton) {
  setAxisAngle(skeleton, "LeftUpperArm",  _AXIS_Z, -ARM_REST_ANGLE);
  setAxisAngle(skeleton, "RightUpperArm", _AXIS_Z,  ARM_REST_ANGLE);
}

export function driveShowcase(skeleton: THREE.Skeleton, t: number) {
  const lt = t % LOOP_DURATION;
  resetDriven(skeleton);
  applyRestPose(skeleton);

  // ── Segment 1 (0–3s): both arms raise to T-pose and back ──
  if (lt < 3) {
    const u = lt / 3;
    const lift = bell(u);
    setAxisAngle(skeleton, "LeftUpperArm",  _AXIS_Z, -ARM_REST_ANGLE * (1 - lift));
    setAxisAngle(skeleton, "RightUpperArm", _AXIS_Z,  ARM_REST_ANGLE * (1 - lift));

  // ── Segment 2 (3–5s): head turn left → right → centre ──
  } else if (lt < 5) {
    const u = (lt - 3) / 2;
    const turn = Math.sin(u * 2 * Math.PI) * 0.6;
    setAxisAngle(skeleton, "Neck", _AXIS_Y, turn);
    setAxisAngle(skeleton, "Head", _AXIS_Y, turn * 0.4);

  // ── Segment 3 (5–8s): gentle hip sway ──
  } else if (lt < 8) {
    const u = (lt - 5) / 3;
    const sway = Math.sin(u * 3 * Math.PI) * 0.18;
    setAxisAngle(skeleton, "Spine", _AXIS_Z, sway);
    setAxisAngle(skeleton, "Chest", _AXIS_Z, -sway * 0.4);

  // ── Segment 4 (8–11s): jaw open → close ──
  } else {
    const u = (lt - 8) / 3;
    setAxisAngle(skeleton, "Jaw", _AXIS_X, bell(u) * 0.35);
  }

  // ── Breathing underlay — runs every frame across all segments ──
  const breathe = Math.sin(t * 2 * Math.PI * BREATHE_HZ) * BREATHE_AMP;
  const spine = skeleton.getBoneByName("Spine");
  if (spine) {
    _SCRATCH_Q.setFromAxisAngle(_AXIS_X, breathe);
    spine.quaternion.multiply(_SCRATCH_Q);
  }
  const chest = skeleton.getBoneByName("Chest");
  if (chest) {
    _SCRATCH_Q.setFromAxisAngle(_AXIS_X, breathe * 0.5);
    chest.quaternion.multiply(_SCRATCH_Q);
  }
}
