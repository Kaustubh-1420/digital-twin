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

> Upload a photo + enter your height → get a fitted 3D body mesh + 6 body measurements.

**[Live Demo](#)** · **[HF Space](#)**

## Stack
- Body fitting: 4D-Humans (HMR2) → SMPL-X body
- Measurements: plane-slice circumferences + geodesic lengths
- Frontend: Next.js + react-three-fiber
- Backend: Gradio on HuggingFace Spaces (ZeroGPU)

## Status
🚧 In development
