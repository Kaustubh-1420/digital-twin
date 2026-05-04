import * as THREE from "three";

// Showcase loop — drives the sample avatar with a hand-authored animation
// so the landing page shows a moving figure before the visitor uploads anything.
// Bypasses the MediaPipe pose pipeline; sets bone quaternions directly.

const LOOP_DURATION = 17;
const ARM_REST_ANGLE = 1.15; // ~66°, leaves clearance from torso

const _AXIS_X = new THREE.Vector3(1, 0, 0);
const _AXIS_Y = new THREE.Vector3(0, 1, 0);
const _AXIS_Z = new THREE.Vector3(0, 0, 1);

const _DRIVEN_BONES = [
  "LeftUpperArm", "RightUpperArm",
  "LeftLowerArm", "RightLowerArm",
  "LeftHand", "RightHand",
  "Neck", "Head",
  "Spine", "Chest", "UpperChest",
  "Jaw",
];

const _LEFT_FINGERS = [
  "LeftThumbMetacarpal", "LeftThumbProximal", "LeftThumbDistal",
  "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
  "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
  "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
  "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
];
const _RIGHT_FINGERS = _LEFT_FINGERS.map((n) => n.replace("Left", "Right"));

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}
function bell(u: number): number {
  return Math.sin(u * Math.PI); // 0 → 1 → 0
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
  for (const n of _LEFT_FINGERS) {
    const b = skeleton.getBoneByName(n);
    if (b) b.quaternion.identity();
  }
  for (const n of _RIGHT_FINGERS) {
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
    return;
  }

  // ── Segment 2 (3–5s): head turn left → right → centre ──
  // Axis assumption: Neck/Head local Y = vertical → rotation around Y = look L/R.
  if (lt < 5) {
    const u = (lt - 3) / 2;
    const turn = Math.sin(u * 2 * Math.PI) * 0.6; // ±0.6 rad (~34°)
    setAxisAngle(skeleton, "Neck", _AXIS_Y, turn);
    setAxisAngle(skeleton, "Head", _AXIS_Y, turn * 0.4);
    return;
  }

  // ── Segment 3 (5–8s): right arm raises to T-pose, hand waves "hi" ──
  // Wave = wrist pivot (RightHand around Y), forearm stays still.
  // Arm does NOT lower at end — Seg 4 inherits T-pose state.
  if (lt < 8) {
    const u = (lt - 5) / 3;
    let armAngle = ARM_REST_ANGLE;
    let waveAngle = 0;
    if (u < 0.3) {
      const ru = u / 0.3;
      armAngle = ARM_REST_ANGLE * (1 - smooth(ru)); // rest → T-pose
    } else {
      armAngle = 0; // hold at T-pose for the rest of the segment
      const wu = (u - 0.3) / 0.7;
      waveAngle = Math.sin(wu * 4 * Math.PI) * 0.6; // 4 cycles, ±0.6 rad wrist swing
    }
    setAxisAngle(skeleton, "RightUpperArm", _AXIS_Z, armAngle);
    setAxisAngle(skeleton, "RightHand",     _AXIS_Y, waveAngle);
    return;
  }

  // ── Segment 4 (8–11s): right hand open → curl → open, palm facing camera ──
  // Forearm twist (RightLowerArm X-axis) rotates palm from down (bind) to forward.
  // Arm lowers back to rest in last 20% of segment.
  if (lt < 11) {
    const u = (lt - 8) / 3;
    let armAngle = 0;
    if (u > 0.8) {
      const lu = (u - 0.8) / 0.2;
      armAngle = ARM_REST_ANGLE * smooth(lu); // T-pose → rest
    }
    setAxisAngle(skeleton, "RightUpperArm", _AXIS_Z, armAngle);
    // Twist forearm so palm faces camera during curl. Sign empirical.
    setAxisAngle(skeleton, "RightLowerArm", _AXIS_X, Math.PI / 2);

    const curl = bell(u) * 1.0;
    for (const name of _RIGHT_FINGERS) {
      if (name.includes("Thumb")) continue;
      setAxisAngle(skeleton, name, _AXIS_Y, curl);
    }
    setAxisAngle(skeleton, "RightThumbProximal", _AXIS_Y, curl * 0.5);
    setAxisAngle(skeleton, "RightThumbDistal",   _AXIS_Y, curl * 0.5);
    return;
  }

  // ── Segment 5 (11–14s): gentle hip sway via spine lean ──
  // Axis: Spine local Z = lateral lean (sign empirical).
  if (lt < 14) {
    const u = (lt - 11) / 3;
    const sway = Math.sin(u * 3 * Math.PI) * 0.18; // 1.5 cycles, ~10° amplitude
    setAxisAngle(skeleton, "Spine", _AXIS_Z, sway);
    setAxisAngle(skeleton, "Chest", _AXIS_Z, -sway * 0.4); // counter-rotate
    return;
  }

  // ── Segment 6 (14–17s): jaw open → close ──
  // Jaw axis confirmed by poseSolver.driveJawEyes — rotation.x opens mouth.
  const u = (lt - 14) / 3;
  const open = bell(u) * 0.35; // ~20° max
  setAxisAngle(skeleton, "Jaw", _AXIS_X, open);
}
