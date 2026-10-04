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

## Usage

```bash
bash tools/shape_eval/setup_local.sh      # clone PyMAF-X + assets into /tmp (needs HF token)
bash tools/shape_eval/fetch_stress.sh     # 6 Unsplash stress photos -> stress/ (gitignored)
.venv/bin/python -u tools/shape_eval/diag.py \
  frontend/public/samples/sample-{female,male}.jpg tools/shape_eval/stress/*.jpg
```

`diag.py` saves the exact 224×224 crop each image is fed as (`stress/out/`),
and prints keypoint visibility and per-stage betas.

`fetch_live_glbs.sh` + `solve_betas.py` recover β from production GLBs without
running inference: with zero pose the mesh is `s·(T + S·β) + t`, linear in
`(s, sβ, t)`. Fit on non-hand vertices — the `smplx` default mean hand pose is
not linear and leaves a ~7.5 mm residual on the full mesh.

Local env notes: Python 3.11 `.venv`; `mediapipe==0.10.21` (1.x aborts on
macOS initialising Metal, even with the CPU delegate).
