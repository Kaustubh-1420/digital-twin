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
