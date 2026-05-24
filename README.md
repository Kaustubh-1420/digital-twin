# digital-twin

**[Live demo](https://digital-twin-ai.vercel.app)** · **[HF Space](https://huggingface.co/spaces/Kaustubh1420/digital-twin)**

Upload a photo and your height. PyMAF-X fits an SMPL-X body model to your proportions, exports a personalized skinned mesh, and streams it to the browser — where it mirrors your movements in real time via webcam.

The avatar is fitted to your estimated body shape, not a generic template. Body pose, hands (30 joints), jaw, and facial expressions are all driven client-side at ~30 fps with no server round-trip after the initial fit.

**[Live demo](https://frontend-chi-cyan-49.vercel.app)** · **[HF Space](https://huggingface.co/spaces/Kaustubh1420/digital-twin)**

---

## How it works

**Body fitting**
PyMAF-X runs on the uploaded photo and estimates SMPL-X shape parameters (`betas`). These are passed directly — no SMPLify conversion step. A T-pose mesh is generated from the betas, scaled to the user-provided height, and exported as a skinned GLB with VRM humanoid bone names and LBS weights.

The betas are a monocular estimate from one image. Relative proportions (shoulder-to-hip ratio, limb lengths) are typically well-captured; absolute girth measurements depend on photo quality and pose.

**Measurements**
Six measurements are extracted from the scaled T-pose mesh:

| Measurement | Method |
|---|---|
| Chest | Ellipse approximation at collar height (arm-safe) |
| Waist | Plane-slice perimeter at mid-torso |
| Hip | Plane-slice perimeter at hip joint level |
| Shoulder width | Vertex-based acromion-to-acromion estimate |
| Inseam | Joint chain: hip → knee → ankle |
| Arm length | Joint chain: shoulder → elbow → wrist |

**Real-time driving**
Three MediaPipe models run in parallel (WASM, client-side) via a shared requestAnimationFrame loop:

- **PoseLandmarker** — 33 body landmarks → 22 bone rotations via swing decomposition in parent-local space. One Euro Filter on landmarks, quaternion SLERP on bones.
- **HandLandmarker** — 21 landmarks per hand → 15 finger bones per side. Per-bone REST directions derived lazily from the GLB bind pose via WeakMap cache.
- **FaceLandmarker** — 52 ARKit blendshape scores. Jaw and eye gaze driven as bone rotations directly. Remaining scores passed through a pre-computed (52×100) ARKit→SMPL-X expression matrix (`bs2exp.npy`) to drive 100 expression morph targets baked into the GLB. A 30-frame neutral baseline is captured on webcam start to zero out ambient eye/brow bias from webcam angle.

The forearm twist redistribution splits pronation/supination between the lower arm and wrist bones to reduce the wrist-knot artifact inherent in single-forearm-bone SMPL-X skeletons.

---

## Stack

| Layer | Tech |
|---|---|
| Body fitting | PyMAF-X (SMPL-X native betas output) |
| Measurements | Plane-slice perimeter + Ramanujan ellipse approx |
| GLB export | Raw GLTF binary — 55 joints, LBS weights, 100 morph targets |
| Backend | Gradio on HF Spaces ZeroGPU |
| Frontend | Next.js 16 + react-three-fiber on Vercel |
| Pose tracking | MediaPipe PoseLandmarker (WASM, client-side) |
| Hand tracking | MediaPipe HandLandmarker (30 joints, swing decomp) |
| Face tracking | MediaPipe FaceLandmarker + ARKit→SMPL-X linear mapping |

---

## Known limitations

- **Betas from monocular image** — body shape is estimated, not measured. Works best with a clear standing photo in fitted clothing.
- **No teeth, tongue, or mouth interior** — SMPL-X models the outer body surface only.
- **Eyelid blink** — SMPL-X expression PCA does not encode eyelid closure; blink is not visible.
- **Fist closure** — MediaPipe hallucinates fingertip positions during full fist occlusion. ~50% closure is the practical limit.
- **Head turn in mirror mode** — deferred; requires a 3-point nose+ear basis for the neck, same as the hand palm basis.

---

## Local setup

```bash
# Backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place SMPL-X models at models/smplx/SMPLX_NEUTRAL.npz
# Place PyMAF-X checkpoint at /tmp/PyMAF-X/data/pretrained_model/

python app.py

# Frontend
cd frontend
npm install --legacy-peer-deps
npm run dev
```

The frontend expects the Gradio backend at the URL configured in `frontend/lib/api.ts`.

---

## Repo structure

```
backend/
  measurements.py       — 6 anthropometric measurements from T-pose mesh
  scale.py              — height calibration
  pymafx_backend.py     — PyMAF-X inference wrapper
  export_glb.py         — skinned GLB: 55 joints, VRM bones, 100 expression morph targets
app.py                  — Gradio app (HF Spaces entry point)
frontend/
  components/
    AvatarCanvas.tsx    — Three.js scene, skeleton driving, camera system, face morph targets
  hooks/
    usePoseLandmarker.ts — MediaPipe rAF loop (pose + hands + face, parallel Promise.all load)
  lib/
    poseSolver.ts       — swing decomposition, forearm twist redistribution, jaw/eye bone drive
    exprMapping.ts      — 52×100 ARKit→SMPL-X expression matrix
    oneEuroFilter.ts    — One Euro Filter + batch landmark smoother
```
