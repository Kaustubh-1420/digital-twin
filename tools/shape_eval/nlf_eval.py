"""Shape acceptance test for NLF (SMPL-X betas straight from the model).

The multi-person TorchScript hardcodes cuda:0 in its YOLO detector, so this
needs an NVIDIA GPU (Colab: tools/shape_eval/nlf_colab.ipynb).

Prints, per image: people detected, time, betas; then the pass-criteria pairs
and the full pairwise |Δβ| (L2 over the first 10 betas, same metric as the
PyMAF-X baseline in README.md).

Usage:
  python tools/shape_eval/nlf_eval.py --model nlf_l_multi_0.3.2.torchscript \
      frontend/public/samples/sample-{female,male}.jpg tools/shape_eval/stress/*.jpg --beta-reg 10 1 0.1 0
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torchvision  # noqa: F401  (registers ops the TorchScript model needs)
from torchvision.io import ImageReadMode, decode_image, read_file

PAIRS = [("thin_m", "thin_m2", "same person (noise floor)"),
         ("heavy_m", "thin_m", "heavy vs thin"),
         ("heavy_m", "muscle_m2", "heavy vs muscular")]
BASELINE = {"thin_m-thin_m2": 0.245, "heavy_m-thin_m": 0.399, "heavy_m-muscle_m2": 0.133}


def run(model, paths, beta_reg):
    betas = {}
    for p in paths:
        name = Path(p).stem.removeprefix("sample-")
        img = decode_image(read_file(p), mode=ImageReadMode.RGB).cuda()
        t = time.time()
        with torch.inference_mode():
            pred = model.detect_smpl_batched(img[None], model_name="smplx", beta_regularizer=beta_reg)
        torch.cuda.synchronize()
        dt = time.time() - t
        b_all = pred["betas"][0].cpu()
        if b_all.shape[0] == 0:
            print(f"  {name:14s} {dt:5.1f}s  NO PERSON DETECTED")
            continue
        # Largest person by 2D joint extent (stress photos have one each)
        ext = pred["joints2d"][0].cpu()
        best = int((ext.amax(1) - ext.amin(1)).prod(-1).argmax())
        b = b_all[best].numpy()
        print(f"  {name:14s} {dt:5.1f}s  people={b_all.shape[0]}  nbetas={b.shape[0]}")
        print(f"      β[:10] {np.array2string(b[:10], precision=3, suppress_small=True)}")
        betas[name] = b[:10]
    return betas


def report(betas):
    print("  pass-criteria pairs (PyMAF-X baseline in brackets):")
    for a, b, label in PAIRS:
        if a in betas and b in betas:
            key = f"{a}-{b}"
            print(f"    {key:20s} {np.linalg.norm(betas[a] - betas[b]):.3f}  [{BASELINE[key]}]  {label}")
    names = list(betas)
    if len(names) < 2:
        return
    print("  pairwise |Δβ|:")
    print("    " + " " * 14 + "".join(f"{n[:9]:>10s}" for n in names))
    for a in names:
        print(f"    {a:14s}" + "".join(f"{np.linalg.norm(betas[a] - betas[b]):10.3f}" for b in names))
    print(f"  max {max(np.linalg.norm(betas[a] - betas[b]) for a, b in itertools.combinations(names, 2)):.3f}"
          f"   mean |β| {np.mean([np.linalg.norm(v) for v in betas.values()]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--beta-reg", type=float, nargs="+", default=[10.0])
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    # map_location='cuda', not .to('cuda'): .to() misses tensors the model
    # keeps in plain dict attributes, leaving them on CPU.
    model = torch.jit.load(args.model, map_location="cuda").eval()
    for reg in args.beta_reg:
        print(f"\n=== {Path(args.model).name}  beta_regularizer={reg}")
        report(run(model, args.images, reg))


if __name__ == "__main__":
    main()
