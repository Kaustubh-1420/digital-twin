"""Time NLF's first-call warm-up under different TorchScript executor settings.

Each mode runs in a fresh process (warm-up is per process, and on the Space
the model is loaded per request). Prints load time, per-image times and the
first image's β, so modes can be checked for identical output.

Usage:  python nlf_speed.py --model nlf_l_multi_0.3.2.torchscript img1.jpg img2.jpg img3.jpg
"""
import argparse
import json
import subprocess
import sys
import time

MODES = {
    "default": "",
    "no_profiling": "torch._C._jit_set_profiling_executor(False)",
    "no_optimize": "torch._C._set_graph_executor_optimize(False)",
    "no_fusers": ("torch._C._jit_set_texpr_fuser_enabled(False); "
                  "torch._C._jit_override_can_fuse_on_gpu(False)"),
}


def child(model_path, mode, images):
    import torch
    import torchvision  # noqa: F401
    from torchvision.io import ImageReadMode, decode_image, read_file
    exec(MODES[mode])
    t = time.time()
    model = torch.jit.load(model_path, map_location="cuda").eval()
    torch.cuda.synchronize()
    out = {"mode": mode, "load": round(time.time() - t, 2), "times": [], "beta0": None}
    for p in images:
        img = decode_image(read_file(p), mode=ImageReadMode.RGB).cuda()
        t = time.time()
        with torch.inference_mode():
            pred = model.detect_smpl_batched(img[None], model_name="smplx")
        torch.cuda.synchronize()
        out["times"].append(round(time.time() - t, 2))
        if out["beta0"] is None and pred["betas"][0].shape[0]:
            out["beta0"] = [round(float(v), 3) for v in pred["betas"][0][0][:4]]
    print("RESULT " + json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--child", choices=MODES)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    if a.child:
        return child(a.model, a.child, a.images)
    print(f"images: {a.images}")
    for mode in MODES:
        r = subprocess.run([sys.executable, __file__, "--model", a.model, "--child", mode, *a.images],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line:
            d = json.loads(line[7:])
            print(f"{mode:13s} load {d['load']:5.1f}s  per-image {d['times']}  β[:4] {d['beta0']}")
        else:
            print(f"{mode:13s} FAILED: {(r.stderr or r.stdout).strip().splitlines()[-1][:300]}")


if __name__ == "__main__":
    main()
