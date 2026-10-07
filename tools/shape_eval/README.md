# Shape-model acceptance test

Checks whether a body-shape estimator actually responds to body shape.
"Numbers in a plausible range" is not enough — a constant average body
passes that. PyMAF-X did, for months.

## Pass criteria

1. Two shots of the same person (`thin_m`, `thin_m2`) are clearly closer in β
   than the heaviest and thinnest bodies (`heavy_m`, `thin_m`).
2. The heavy-set and muscular bodies (`heavy_m`, `muscle_m2`) separate.

Baseline to beat — PyMAF-X, final stage:

| pair | \|Δβ\| |
|---|---|
| thin_m vs thin_m2 (same person, noise floor) | 0.25 |
| heavy_m vs thin_m | 0.40 |
| heavy_m vs muscle_m2 | 0.13 |
| max over all 8 photos | 0.71 (random adults ≈ 4.5) |

## Results: NLF (2026-10-07) — PASS

NLF-L v0.3.2 (`nlf_l_multi_0.3.2.torchscript`, `model_name='smplx'`), run on a
Colab T4 via `nlf_colab.ipynb` (`nlf_eval.py`). Default `beta_regularizer=10`:

| pair | NLF | PyMAF-X |
|---|---|---|
| thin_m vs thin_m2 (same person) | 0.258 | 0.245 |
| heavy_m vs thin_m | 4.355 | 0.399 |
| heavy_m vs muscle_m2 | 2.328 | 0.133 |
| max over all 8 photos | 5.09 | 0.71 |
| photo with no person | none detected | body returned |

Heavy vs thin is 17× the same-person gap (PyMAF-X: 1.6×). Only β0 and β1 are
non-zero at the default. Sweep: `beta_regularizer` 1 is near-identical, 0.1
slightly noisier (same person 0.295), 0 breaks down (same person 1.874, mean
|β| 4.4). So NLF reliably pins about two shape directions (size, weight); finer
proportions would have to come from a fit to silhouette/keypoints.

The multi-person TorchScript can't run on CPU (its YOLO detector hardcodes
`cuda:0`). Load it with `torch.jit.load(path, map_location='cuda')`: `.to('cuda')`
misses tensors held in dict attributes, and under ZeroGPU the load has to happen
inside `@spaces.GPU`.

## Usage

```bash
bash tools/shape_eval/setup_local.sh      # clone PyMAF-X + assets into /tmp (needs HF token)
bash tools/shape_eval/fetch_stress.sh     # 6 Unsplash stress photos -> stress/ (gitignored)
.venv/bin/python -u tools/shape_eval/diag.py \
  frontend/public/samples/sample-{female,male}.jpg tools/shape_eval/stress/*.jpg
```

`diag.py` saves the exact 224×224 crop each image is fed as (`stress/out/`),
and prints keypoint visibility and per-stage betas.

PyMAF-X was removed from the app after the NLF pass. `setup_local.sh` and
`diag.py` (the PyMAF-X baseline) need the repo at commit `09d592c`.

`fetch_live_glbs.sh` + `solve_betas.py` recover β from production GLBs without
running inference: with zero pose the mesh is `s·(T + S·β) + t`, linear in
`(s, sβ, t)`. Fit on non-hand vertices — the `smplx` default mean hand pose is
not linear and leaves a ~7.5 mm residual on the full mesh.

Local env notes: Python 3.12.12 `.venv` via uv, matching the Space (ZeroGPU:
Python 3.12, torch 2.9.1). Install `requirements.txt` with overrides
`mediapipe==0.10.21` (1.x aborts on macOS initialising Metal, even with the
CPU delegate) and `gradio==6.13.0` (the Space's `sdk_version`). β output is
identical to the old 3.11 / torch 2.14 env on all 8 photos.
