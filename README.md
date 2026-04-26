---
title: Digital Twin
emoji: 🧍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.13.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# digital-twin

Upload a photo and your height. Get a 3D body mesh fitted to your proportions, body measurements, and a webcam mirror where the avatar follows your movements in real time.

**[Live demo](https://frontend-chi-cyan-49.vercel.app)** · **[HF Space](https://huggingface.co/spaces/Kaustubh1420/digital-twin)**

---

## How it works

**Body fitting**
PyMAF-X runs on the uploaded photo and estimates SMPL-X shape parameters (`betas`) for your specific body proportions — not a generic avatar. The betas are a monocular estimate from one image, so absolute accuracy depends on photo quality and pose, but relative proportions (shoulder-to-hip ratio, limb lengths) are typically well-captured.

**Measurements**
With the betas, a T-pose SMPL-X mesh is generated and scaled to the user-provided height. Six measurements are extracted:

| Measurement | Method |
|---|---|
| Chest | Ellipse approximation at collar height (arm-safe) |
| Waist | Plane-slice perimeter at mid-torso |
| Hip | Plane-slice perimeter at hip joint level |
| Shoulder width | Vertex-based acromion-to-acromion distance |
| Inseam | Joint chain: hip → knee → ankle |
| Arm length | Joint chain: shoulder → elbow → wrist |

The mesh + skeleton are exported as a skinned GLB with VRM humanoid bone names.

**Real-time driving**
In the browser, MediaPipe PoseLandmarker and HandLandmarker run client-side (WASM) via the webcam. Landmark positions are converted to bone rotations using swing decomposition in parent-local space — the same approach used in VTuber software. One Euro Filter smooths landmarks; quaternion SLERP smooths bone transitions. The avatar mirrors your pose at ~30 fps with no server round-trip.

---

## Stack

| Layer | Tech |
|---|---|
| Body fitting | PyMAF-X (SMPL-X native output) |
| Measurements | trimesh plane-slice + Ramanujan ellipse approx |
| GLB export | Raw GLTF binary (no pygltflib) |
| Backend | Gradio on HF Spaces ZeroGPU |
| Frontend | Next.js 16 + react-three-fiber on Vercel |
| Pose tracking | MediaPipe PoseLandmarker (WASM, client-side) |
| Hand tracking | MediaPipe HandLandmarker (30 joints, swing decomp) |

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
  measurements.py     — 6 anthropometric measurements from T-pose mesh
  scale.py            — height calibration
  pymafx_backend.py   — PyMAF-X inference wrapper
  export_glb.py       — skinned GLB with VRM bones + expression morph targets
app.py                — Gradio app (HF Spaces entry point)
frontend/
  components/
    AvatarCanvas.tsx  — Three.js scene, skeleton driving, camera system
  hooks/
    usePoseLandmarker.ts — MediaPipe rAF loop (pose + hands + face)
  lib/
    poseSolver.ts     — swing decomposition, bone quaternions
    oneEuroFilter.ts  — One Euro Filter + batch landmark smoother
```
