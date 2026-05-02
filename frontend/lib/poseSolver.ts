import * as THREE from "three";
import type { PoseLandmarks, HandLandmarks } from "@/hooks/usePoseLandmarker";

const DEBUG_POSE  = false;  // body logs — off during hand calibration
const DEBUG_HANDS = true;   // hand logs — log obsWorld/obsLocal for key finger bones
let _dbgFrame = 0;
let _dbgHandFrame = 0;

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

const VISIBILITY_THRESHOLD = 0.35;

// ── Rest directions ───────────────────────────────────────────────────────────
// MediaPipe world landmarks are Y-UP (hip-centred, metric). No coordinate
// transform needed — compare restDir to observed direction directly.
// Using clean cardinal directions to avoid compounding small Z errors.
const R = (x: number, y: number, z: number) =>
  new THREE.Vector3(x, y, z).normalize();

const REST: Record<string, THREE.Vector3> = {
  Spine:         R( 0,     1,     0),
  // Calibrated from neutral-pose obsLocal across upright sitting frames.
  // (0,0.75,0.65) ≈ average obsLocal when spine lean is minimal.
  Neck:          R( 0,  0.75,  0.65),
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
  { name: "Neck",       from: MP.NOSE,     to: MP.NOSE,          vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER, MP.NOSE] },

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
  { name: "Neck",       from: MP.NOSE,     to: MP.NOSE,          vis: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER, MP.NOSE] },

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

const _v1 = new THREE.Vector3();   // scratch: observed in parent-local space
const _q1 = new THREE.Quaternion(); // scratch: parentWorldQ.inverse
const _localQ = new THREE.Quaternion();
const _parentWorldQ = new THREE.Quaternion();
// Ear-twist scratch (avoids GC in hot loop)
const _earVec   = new THREE.Vector3();
const _earPerp  = new THREE.Vector3();
const _swingRight = new THREE.Vector3();
const _twistQ   = new THREE.Quaternion();

// Hand bone driving scratch (separate from body/finger scratch to avoid conflicts)
const _hForward = new THREE.Vector3();
const _hKnuckle = new THREE.Vector3();
const _hPalmN   = new THREE.Vector3();
const _hSide    = new THREE.Vector3();
const _hMat     = new THREE.Matrix4();
const _hWorldQ  = new THREE.Quaternion();
const _hParentQ = new THREE.Quaternion();
const _hLocalQ  = new THREE.Quaternion();

// Forearm-twist redistribution: bleed a fraction of wrist twist (around forearm long axis)
// out of the Hand bone and into the LowerArm bone, to relieve LBS mesh collapse at the wrist.
// driveSkeleton writes _pureArmWorldQ (swing-only, before twist injection) so driveHands can
// extract the TRUE total pronation twist rather than the residual left after last frame's bleed.
// driveHands writes _desiredForearmTwist; driveSkeleton reads it when driving LowerArm.
const FOREARM_TWIST_FRACTION = 0.5;
const _desiredForearmTwist = {
  Left:  new THREE.Quaternion(),
  Right: new THREE.Quaternion(),
};
// Pure-swing LowerArm world Q (no forearm twist) — written by driveSkeleton, read by driveHands.
const _pureArmWorldQ = {
  Left:  new THREE.Quaternion(),
  Right: new THREE.Quaternion(),
};
const _qTwistX  = new THREE.Quaternion();
const _qFaTwist = new THREE.Quaternion();
const _identityQ = new THREE.Quaternion();

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
  // spineDir = direction from hip origin to shoulderMid.
  const spineDir = shoulderMid.clone().normalize();
  // neckDir = direction from shoulderMid to nose — drives head turn/tilt.
  const nosePos  = new THREE.Vector3(src[MP.NOSE].x, src[MP.NOSE].y, src[MP.NOSE].z);
  const neckDir  = nosePos.clone().sub(shoulderMid).normalize();

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
    if (SPINE_BONES.has(cfg.name))      observed = spineDir;
    else if (NECK_BONES.has(cfg.name))  observed = neckDir;
    else                                observed = lmDir(src, cfg.from, cfg.to);

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

    // Ear-based twist for Neck: adds Y-axis (look left/right) rotation that
    // swing-only decomp loses because nosePos arc from shoulderMid is compressed.
    if (NECK_BONES.has(cfg.name) &&
        landmarkOk(src, MP.LEFT_EAR) && landmarkOk(src, MP.RIGHT_EAR)) {
      // earVec = leftEar - rightEar in parent-local space
      _earVec.set(
        src[MP.LEFT_EAR].x - src[MP.RIGHT_EAR].x,
        src[MP.LEFT_EAR].y - src[MP.RIGHT_EAR].y,
        src[MP.LEFT_EAR].z - src[MP.RIGHT_EAR].z,
      ).normalize().applyQuaternion(_q1);

      // Project onto plane perpendicular to neckDirLocal (_v1)
      const earDotNeck = _earVec.dot(_v1);
      _earPerp.copy(_earVec).addScaledVector(_v1, -earDotNeck);
      const earPerpLen = _earPerp.length();
      if (earPerpLen > 0.1) {
        _earPerp.divideScalar(earPerpLen);
        // Bone's "right" axis after swing
        _swingRight.set(1, 0, 0).applyQuaternion(_localQ);
        _twistQ.setFromUnitVectors(_swingRight, _earPerp);
        _localQ.premultiply(_twistQ);
      }
    }

    // Snapshot pure-swing LowerArm world Q (before twist injection) so driveHands can
    // extract the true total pronation against an uncontaminated parent frame.
    if (cfg.name === "LeftLowerArm")  _pureArmWorldQ.Left.copy(_parentWorldQ).multiply(_localQ);
    if (cfg.name === "RightLowerArm") _pureArmWorldQ.Right.copy(_parentWorldQ).multiply(_localQ);

    // Post-multiply forearm-twist target into LowerArm local rotation (around its X = long axis).
    if (cfg.name === "LeftLowerArm")  _localQ.multiply(_desiredForearmTwist.Left);
    if (cfg.name === "RightLowerArm") _localQ.multiply(_desiredForearmTwist.Right);

    if (!_prevBoneQ.has(cfg.name)) _prevBoneQ.set(cfg.name, new THREE.Quaternion());
    _prevBoneQ.get(cfg.name)!.copy(_localQ);

    bone.quaternion.slerp(_localQ, 0.5);
    bone.updateWorldMatrix(false, false);
  }
}

export function resetSkeletonDriverState(): void {
  _prevBoneQ.clear();
  _dbgFrame = 0;
}

// ── Hand driving ──────────────────────────────────────────────────────────────

// MediaPipe HandLandmarker landmark indices (0–20 per hand)
const HMP = {
  WRIST:      0,
  THUMB_CMC:  1, THUMB_MCP:  2, THUMB_IP:  3, THUMB_TIP:  4,
  INDEX_MCP:  5, INDEX_PIP:  6, INDEX_DIP:  7, INDEX_TIP:  8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP:  13, RING_PIP:  14, RING_DIP:  15, RING_TIP:  16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
} as const;

type FingerBoneCfg = { name: string; from: number; to: number };

// Structural map only — REST directions are derived from the GLB's actual bind pose
// (bone.children[0].position) at first lookup, not hardcoded. See _getFingerRest.
const _FINGER_TEMPLATE: Array<{
  finger: string; from: number; to: number; suffix: string;
}> = [
  { finger: "Thumb",  from: HMP.THUMB_CMC,  to: HMP.THUMB_MCP,  suffix: "Metacarpal"   },
  { finger: "Thumb",  from: HMP.THUMB_MCP,  to: HMP.THUMB_IP,   suffix: "Proximal"     },
  { finger: "Thumb",  from: HMP.THUMB_IP,   to: HMP.THUMB_TIP,  suffix: "Distal"       },
  { finger: "Index",  from: HMP.INDEX_MCP,  to: HMP.INDEX_PIP,  suffix: "Proximal"     },
  { finger: "Index",  from: HMP.INDEX_PIP,  to: HMP.INDEX_DIP,  suffix: "Intermediate" },
  { finger: "Index",  from: HMP.INDEX_DIP,  to: HMP.INDEX_TIP,  suffix: "Distal"       },
  { finger: "Middle", from: HMP.MIDDLE_MCP, to: HMP.MIDDLE_PIP, suffix: "Proximal"     },
  { finger: "Middle", from: HMP.MIDDLE_PIP, to: HMP.MIDDLE_DIP, suffix: "Intermediate" },
  { finger: "Middle", from: HMP.MIDDLE_DIP, to: HMP.MIDDLE_TIP, suffix: "Distal"       },
  { finger: "Ring",   from: HMP.RING_MCP,   to: HMP.RING_PIP,   suffix: "Proximal"     },
  { finger: "Ring",   from: HMP.RING_PIP,   to: HMP.RING_DIP,   suffix: "Intermediate" },
  { finger: "Ring",   from: HMP.RING_DIP,   to: HMP.RING_TIP,   suffix: "Distal"       },
  { finger: "Little", from: HMP.PINKY_MCP,  to: HMP.PINKY_PIP,  suffix: "Proximal"     },
  { finger: "Little", from: HMP.PINKY_PIP,  to: HMP.PINKY_DIP,  suffix: "Intermediate" },
  { finger: "Little", from: HMP.PINKY_DIP,  to: HMP.PINKY_TIP,  suffix: "Distal"       },
];

function _buildFingerCfgs(side: "Left" | "Right"): FingerBoneCfg[] {
  return _FINGER_TEMPLATE.map(t => ({
    name: `${side}${t.finger}${t.suffix}`,
    from: t.from,
    to:   t.to,
  }));
}

// REST = direction this bone points toward its child at bind, in parent-local space.
// All SMPL-X finger bind quaternions are identity (verified 2026-05-02), so the child's
// `.position` normalized IS the parent's local-space bind direction. Tip bones (Distal)
// have no child — they inherit the parent's REST as a continuation-of-curl approximation.
// WeakMap keyed by bone reference: when a new GLB loads, old bones are GC'd and their
// cache entries vanish automatically.
const _fingerRestCache = new WeakMap<THREE.Bone, THREE.Vector3>();

function _getFingerRest(bone: THREE.Bone): THREE.Vector3 {
  const cached = _fingerRestCache.get(bone);
  if (cached) return cached;
  let dir: THREE.Vector3;
  const child = bone.children[0] as THREE.Bone | undefined;
  if (child) {
    dir = child.position.clone().normalize();
  } else {
    // Tip bone (no child): extrapolate the curl arc from grandparent→parent→tip.
    // Bind quats are identity for the SMPL-X hand chain, so all local frames are
    // parallel — vectors compose across them without frame conversion. Naively
    // copying the parent's REST gives Q=identity at straight finger, leaving the
    // tip stuck at bind-pose curl ("hooked fingertip" symptom).
    const parent = bone.parent as THREE.Bone | null;
    const gp = parent?.parent as THREE.Bone | null;
    const restP  = bone.position.lengthSq() > 1e-10 ? bone.position.clone().normalize() : null;
    const restGP = gp?.children[0]?.position.clone().normalize();
    if (restP && restGP) {
      const Q_curl = new THREE.Quaternion().setFromUnitVectors(restGP, restP);
      dir = restP.clone().applyQuaternion(Q_curl);
    } else if (restP) {
      dir = restP;
    } else {
      dir = new THREE.Vector3(1, 0, 0);
    }
  }
  _fingerRestCache.set(bone, dir);
  return dir;
}

const LEFT_FINGER_CFGS  = _buildFingerCfgs("Left");
const RIGHT_FINGER_CFGS = _buildFingerCfgs("Right");

const _LOG_HAND_BONES = new Set([
  "LeftThumbMetacarpal",
  "LeftIndexProximal", "LeftMiddleProximal", "LeftRingProximal", "LeftLittleProximal",
]);

function _driveOneSide(
  bones: Map<string, THREE.Bone>,
  lms: HandLandmarks,
  cfgs: FingerBoneCfg[],
  side: "Left" | "Right",
): void {
  // HandLandmarker world landmarks are Y-down (same as PoseLandmarker, camera-space convention).
  // Flip both Y and Z to match body-space Y-up / away-from-camera +Z convention.
  const src = lms.map(lm => ({ x: lm.x, y: -lm.y, z: -lm.z, visibility: 1 }));

  // ── Drive Hand (wrist) bone from palm landmark frame ────────────────────────
  // Must run BEFORE finger loop so finger bones see correct parentWorldQ.
  // Wrist landmark = src[0], middle_mcp = src[9], index_mcp = src[5], pinky_mcp = src[17].
  const handBone = bones.get(`${side}Hand`);
  if (handBone) {
    // forward: wrist → middle_mcp — the primary finger direction
    _hForward.set(
      src[9].x - src[0].x,
      src[9].y - src[0].y,
      src[9].z - src[0].z,
    ).normalize();

    // knuckleLine: index_mcp - pinky_mcp (same for both hands).
    // Both hands share camera world space — no per-hand sign flip needed.
    // index_mcp is always on the thumb-side, so this vector + hForward gives palmN toward camera.
    _hKnuckle.set(
      src[5].x - src[17].x,
      src[5].y - src[17].y,
      src[5].z - src[17].z,
    );

    // palmNormal = knuckle × forward — perpendicular to palm surface, points toward camera.
    // Empirically derived 2026-05-01 from hPalmN log: forward×knuckle gave -Z (away from camera)
    // when user's palm faced camera; knuckle×forward gives +Z (toward camera) as required.
    _hPalmN.crossVectors(_hKnuckle, _hForward).normalize();
    // Gram-Schmidt: sideAxis guaranteed ⊥ to both forward and palmNormal
    _hSide.crossVectors(_hForward, _hPalmN).normalize();

    // DEBUG: log rest-pose local axes once so we know which column = finger direction
    if (_dbgHandFrame === 1 && side === "Left") {
      const q0 = new THREE.Quaternion(); handBone.getWorldQuaternion(q0);
      console.log(`[HandDebug] ${side}Hand REST axes (world):`,
        `X=${fv(new THREE.Vector3(1,0,0).applyQuaternion(q0))}`,
        `Y=${fv(new THREE.Vector3(0,1,0).applyQuaternion(q0))}`,
        `Z=${fv(new THREE.Vector3(0,0,1).applyQuaternion(q0))}`,
      );
    }

    // DEBUG: sanity-check palm frame vectors every 60 frames
    if (DEBUG_HANDS && _dbgHandFrame % 60 === 0) {
      console.log(`[HandDebug] frame=${_dbgHandFrame} ${side} hForward=${fv(_hForward)} hPalmN=${fv(_hPalmN)} hSide=${fv(_hSide)}`);
      // Log arm parent world axes to diagnose asymmetry in local euler angles
      const _armX = new THREE.Vector3(1,0,0).applyQuaternion(_pureArmWorldQ[side]);
      const _armY = new THREE.Vector3(0,1,0).applyQuaternion(_pureArmWorldQ[side]);
      const _armZ = new THREE.Vector3(0,0,1).applyQuaternion(_pureArmWorldQ[side]);
      console.log(`[ArmDebug] frame=${_dbgHandFrame} ${side}LowerArm world axes: X=${fv(_armX)} Y=${fv(_armY)} Z=${fv(_armZ)}`);
    }

    // Build world quaternion from orthonormal hand frame.
    // Right arm REST=R(-1,0,0) makes its local X anti-parallel to hForward.
    // Negate hForward so local X aligns with the arm axis (small local rotation).
    // Also negate hPalmN so dorsum points AWAY from camera (-Z), matching left hand.
    // hSide = (-forward) × (-palmN) = forward × palmN = unchanged.
    if (side === "Right") { _hForward.negate(); _hPalmN.negate(); }
    _hMat.makeBasis(_hForward, _hPalmN, _hSide);
    _hWorldQ.setFromRotationMatrix(_hMat);

    // Extract twist against PURE-SWING LowerArm world Q (written by driveSkeleton this frame,
    // before forearm-twist injection). Using the actual bone's world Q would include last frame's
    // accumulated twist, shrinking the extracted amount each frame and converging to wrong equilibrium.
    _hLocalQ.copy(_pureArmWorldQ[side]).invert().multiply(_hWorldQ);
    _qTwistX.set(_hLocalQ.x, 0, 0, _hLocalQ.w).normalize();
    _qFaTwist.copy(_identityQ).slerp(_qTwistX, FOREARM_TWIST_FRACTION);
    _desiredForearmTwist[side].copy(_qFaTwist);

    // Back-project hand local Q from world target against the ACTUAL (with-twist) parent.
    // This guarantees world hand = _hWorldQ exactly every frame — no compensation term needed.
    _hParentQ.identity();
    if (handBone.parent) handBone.parent.getWorldQuaternion(_hParentQ);
    _hLocalQ.copy(_hParentQ).invert().multiply(_hWorldQ);

    // DEBUG: log every hemisphere flip and every-60-frame palmN.z to diagnose balloon twist.
    // palmN.z should be <0 when palm faces camera (dorsum faces away = -Z in body-space convention).
    if (DEBUG_HANDS) {
      const dot = handBone.quaternion.dot(_hLocalQ);
      if (dot < 0) {
        console.log(`[WristTwist] frame=${_dbgHandFrame} ${side} FLIP dot=${dot.toFixed(3)} palmN.z=${_hPalmN.z.toFixed(3)} hPalmN=${fv(_hPalmN)}`);
      }
      if (_dbgHandFrame % 60 === 0) {
        const _eu = new THREE.Euler().setFromQuaternion(_hLocalQ, 'XYZ');
        console.log(`[WristTwist] frame=${_dbgHandFrame} ${side} palmN.z=${_hPalmN.z.toFixed(3)} dot=${dot.toFixed(3)} localEuler=(${(_eu.x*180/Math.PI).toFixed(1)}°,${(_eu.y*180/Math.PI).toFixed(1)}°,${(_eu.z*180/Math.PI).toFixed(1)}°)`);
      }
    }

    // Quaternion hemisphere check: if the new target lies in the opposite hemisphere from the
    // current bone quaternion, slerp would take the LONG way around (the "balloon twist" symptom).
    // Negating all 4 components yields the same rotation but in the matching hemisphere.
    if (handBone.quaternion.dot(_hLocalQ) < 0) {
      _hLocalQ.set(-_hLocalQ.x, -_hLocalQ.y, -_hLocalQ.z, -_hLocalQ.w);
    }
    handBone.quaternion.slerp(_hLocalQ, 0.4);
    handBone.updateWorldMatrix(false, true); // propagate to finger children
  }

  const logHand = DEBUG_HANDS && (_dbgHandFrame % 60 === 0);

  for (const cfg of cfgs) {
    const bone = bones.get(cfg.name);
    if (!bone) continue;

    const from = src[cfg.from];
    const to   = src[cfg.to];
    if (!from || !to) continue;

    const dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
    const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (len < 1e-6) continue;

    // Observed direction in parent-local space (same local-space swing decomp as body)
    _parentWorldQ.identity();
    if (bone.parent) bone.parent.getWorldQuaternion(_parentWorldQ);
    _q1.copy(_parentWorldQ).invert();
    _v1.set(dx/len, dy/len, dz/len).applyQuaternion(_q1);
    const restDir = _getFingerRest(bone);
    _localQ.setFromUnitVectors(restDir, _v1);

    if (logHand && _LOG_HAND_BONES.has(cfg.name)) {
      console.log(
        `[HandSolver] frame=${_dbgHandFrame} ${cfg.name.padEnd(22)}`,
        `obsWorld=(${(dx/len).toFixed(3)},${(dy/len).toFixed(3)},${(dz/len).toFixed(3)})`,
        `obsLocal=${fv(_v1)}`,
        `rest=${fv(restDir)}`,
        `→ ${fq(_localQ)}`,
      );
    }

    bone.quaternion.slerp(_localQ, 0.3);
    bone.updateWorldMatrix(false, false);
  }
}

export function driveHands(
  skeleton: THREE.Skeleton,
  leftHandLms: HandLandmarks | null,
  rightHandLms: HandLandmarks | null,
): void {
  const bones: Map<string, THREE.Bone> = new Map();
  skeleton.bones.forEach(b => bones.set(b.name, b));

  const hasHands = (leftHandLms && leftHandLms.length >= 21) || (rightHandLms && rightHandLms.length >= 21);
  if (hasHands) _dbgHandFrame++;

  if (leftHandLms  && leftHandLms.length  >= 21) _driveOneSide(bones, leftHandLms,  LEFT_FINGER_CFGS,  "Left");
  if (rightHandLms && rightHandLms.length >= 21) _driveOneSide(bones, rightHandLms, RIGHT_FINGER_CFGS, "Right");
}

