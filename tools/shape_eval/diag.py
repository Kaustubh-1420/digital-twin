"""H-A vs H-B: does PyMAF-X actually see the person, and does pose vary while shape doesn't?

Replicates pymafx_backend.infer() step by step, keeping intermediates:
  - keypoints + whether the no-person fallback fired
  - the exact 224x224 crop fed to the network (saved as PNG)
  - final-stage betas and every pose-like output, compared across images
"""
import os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stress', 'out'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, 'backend'))

import cv2, torch
import pymafx_backend as pb

# mediapipe 1.x on macOS aborts initialising a Metal service; force the CPU delegate.
import functools
from mediapipe.tasks import python as _mpt
_mpt.BaseOptions = functools.partial(_mpt.BaseOptions, delegate=_mpt.BaseOptions.Delegate.CPU)

images = sys.argv[1:] or [f'{REPO}/frontend/public/samples/sample-{g}.jpg' for g in ('female', 'male')]
model, device = pb.load_model()

results = {}
for path in images:
    name = os.path.splitext(os.path.basename(path))[0]
    img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    kp = pb.detect_body_keypoints(img_rgb)
    vis = kp[:, 2] > 0.3
    batch = pb._make_batch(img_rgb, kp)

    # undo ImageNet normalisation to see exactly what the network sees
    from core import constants
    t = batch['img_body'][0].numpy()
    mean = np.array(constants.IMG_NORM_MEAN)[:, None, None]
    std = np.array(constants.IMG_NORM_STD)[:, None, None]
    crop = np.clip((t * std + mean).transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{OUT}/crop_{name}.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    cwd = os.getcwd(); os.chdir(pb.PYMAFX_DIR)
    try:
        with torch.no_grad():
            preds, _ = model(batch)
    finally:
        os.chdir(cwd)

    stages = preds['mesh_out']
    final = stages[-1]
    results[name] = {k: v.detach().cpu().numpy() for k, v in final.items() if isinstance(v, torch.Tensor)}
    print(f'\n=== {name}  image {w}x{h}')
    print(f'  keypoints visible >0.3: {vis.sum()}/{len(kp)}   x[{kp[vis,0].min():.0f},{kp[vis,0].max():.0f}] y[{kp[vis,1].min():.0f},{kp[vis,1].max():.0f}]')
    print(f'  stages: {len(stages)}; betas per stage:')
    for i, s in enumerate(stages):
        print(f'    [{i}] {np.round(s["pred_shape"][0].cpu().numpy(), 3)}')

if len(results) >= 2:
    a, b = list(results)[:2]
    print(f'\n=== final-stage outputs, {a} vs {b}  (relative diff = |a-b| / mean(|a|,|b|))')
    for k in sorted(results[a]):
        va, vb = results[a][k], results[b][k]
        if va.shape != vb.shape or va.size == 0:
            continue
        rel = np.linalg.norm(va - vb) / (0.5 * (np.linalg.norm(va) + np.linalg.norm(vb)) + 1e-9)
        print(f'  {k:28s} shape {str(va.shape):18s} rel diff {rel:.3f}')
print(f'\ncrops saved to {OUT}/crop_*.png')
