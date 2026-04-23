# Work Log

## 2026-04-19
- Scaffolded repo structure (backend, frontend, models, data dirs)
- Set up Python 3.11 venv, installed 4D-Humans + SMPL-X deps
- Downloaded SMPL-X models, ran 4D-Humans demo successfully
- Implemented smpl_to_smplx() converter (rotmat → axis-angle, betas passthrough)
- Implemented T-pose normalization (zero body_pose, keep betas)
- Implemented measurements.py — 6 measurements (chest ellipse + waist/hip plane-slice + joint-chain lengths)
- Implemented scale_to_height() calibration

## 2026-04-23
- Critique review: identified 3 bugs in existing code (plane_slice undercount, scale_to_height missing joints, chest ellipse X-centering)
- Researched SMPL→SMPL-X shape transfer: PyMAF-X (direct SMPL-X output) vs HMR2 + transfer_model tool
- Researched real-time avatar driving: Kalidokit-style PoseSolver approach confirmed viable (VTuber stack)
- Goal expanded: static measurements → real-time personalized digital twin (webcam mirroring)
- Updated task list, memory, session workflow
- Next: fix 3 bugs → try PyMAF-X setup → validate on 10 photos

## 2026-04-24 (session 3)
- Fixed Bug 1: _plane_slice_perimeter now uses e.length() — closing segment no longer missing
- Fixed Bug 2: scale_to_height() now accepts + returns joints, scaled by same factor; test_inference.py now calls it
- Fixed Bug 3: chest ellipse centers mesh at X=0 before masking — robust to global_orient residual
- Added --image and --height CLI args to test_inference.py
- Ran full pipeline successfully: 175cm calibration exact, measurements in plausible range
- PyMAF-X setup started: cloned repo, identified pretrained model download needed from pan.bnu.edu.cn (web UI, manual download required)
- pytorch3d (renderer only) and openpifpaf (detection only) can be mocked/skipped for our inference wrapper
- Next: user downloads PyMAF-X checkpoint + partial_mesh files → install deps → write inference wrapper

## 2026-04-23 (session 4)
- Task 9 complete: PyMAF-X fully set up
  - Moved downloaded checkpoint + partial_mesh files into /tmp/PyMAF-X/data/
  - Fixed numpy 2.x incompatibility (twodim_base removed) in maf_extractor.py
  - Installed boto3, mediapipe, scikit-learn into venv
  - Wrote backend/pymafx_backend.py: MediaPipe Tasks API for keypoints → PyMAF-X forward → native SMPL-X betas
  - Updated test_inference.py with --backend flag (pymafx default, hmr2 fallback)
- Task 10 complete: validated on 10 diverse photos
  - 10/10 PASS, no crashes, all measurements in human range
  - 2/10 images: MediaPipe fallback bbox used (still ran fine)
  - Wrote backend/validate_10.py for batch testing
- Task 11 complete: Gradio app built and tested locally
  - app.py: image + height → PyMAF-X → T-pose → measurements + GLB avatar
  - gr.Model3D for avatar preview in browser
  - @spaces.GPU(duration=120) for ZeroGPU
  - setup.sh: clones PyMAF-X, patches numpy, downloads assets from private HF dataset via HF_TOKEN
  - requirements.txt for HF Spaces
  - UI works end-to-end, user reviewed it locally

## 2026-04-23 (session 5) — Task 12: HF Spaces deploy
- Created HF account (Kaustubh1420), access token (write), logged in via `hf auth login`
- Created private dataset: Kaustubh1420/smplx-measure-assets
- Uploaded all assets to dataset:
  - PyMAF-X checkpoint (~400MB), all smplx partial_mesh files
  - SMPLX_NEUTRAL.npz + .pkl, SMPLX_NEUTRAL_2020.npz
  - smpl/SMPL_NEUTRAL.pkl, MANO_RIGHT.pkl, model_transfer/* , FLAME2020/*
  - data/J_regressor_extra.npy, smpl_mean_params.npz, flame_downsampling.npy
- Renamed project from smplx-measure → digital-twin throughout codebase
- Created Space: Kaustubh1420/digital-twin (Gradio, ZeroGPU)
- Fixed multiple Python 3.13 compat issues in pymafx_backend.py:
  - Mocked all stdlib modules removed in 3.13 (cgi, imp, aifc, etc.)
  - Added proper numpy-backed chumpy shim (chumpy.Ch as ndarray subclass, chumpy.ch submodule)
- Fixed setup.sh to download all required data files from private dataset
- Fixed app.py: _run_setup() triggers setup.sh if /tmp/PyMAF-X missing, SMPLX path auto-detected
- Added packages.txt: libgles2, libegl1 (MediaPipe system deps)
- Added scikit-image to requirements.txt
- Status: Space is UP, setup.sh runs cleanly, MediaPipe works (17/17 landmarks)
- Blocked on: chumpy.ch submodule pickle deserialization — fix pushed, awaiting restart
- Next: verify clean load after chumpy.ch fix → if more errors, keep fixing → Task 13

## 2026-04-24 (session 6) — Task 12 complete, GitHub remote set up
- Fixed chumpy shim (series of fixes across multiple deploys):
  - _Ch: plain Python class (not ndarray subclass) with __setstate__ reading 'x' key
  - Added MetaPathFinder import hook to catch any chumpy.* submodule at runtime
  - _Select.__setstate__: reconstructs a.ravel()[idxs].reshape(preferred_shape) — confirmed format by inspecting MANO_RIGHT.pkl with real chumpy
  - Pre-registered known submodules: chumpy.ch, reordering, utils, optimization, linalg, logic, indexed_inputs, check_derivatives
- Fixed missing files: MANO_LEFT.pkl (copy of RIGHT in setup.sh), smpl_vert_segmentation.json (uploaded from GarmentCode repo)
- Model loads cleanly, inference runs end-to-end on HF Spaces ZeroGPU ✓
- Set up GitHub remote: github.com/Kaustubh-1420/digital-twin (private, SSH auth)
- All changes committed and pushed to GitHub
- Next: Task 13 — scaffold Next.js frontend
