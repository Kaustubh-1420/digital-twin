#!/bin/bash
# Live stress test: posts 4 stress photos to the production /api/predict
# (Vercel -> HF Space, same path as the site) at a fixed height of 175 cm,
# and prints each photo's measurements and wall time. ~4 GPU runs.
DIR="$(cd "$(dirname "$0")" && pwd)/stress"
URL=https://digital-twin-ai.vercel.app/api/predict
for name in heavy_m thin_m thin_m2 muscle_m2; do
  start=$(date +%s)
  out=$(curl -s -m 120 -X POST "$URL" -F "image=@$DIR/$name.jpg;type=image/jpeg" -F "heightCm=175")
  secs=$(( $(date +%s) - start ))
  echo "=== $name (${secs}s)"
  printf '%s' "$out" | python3 -c '
import sys, json
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("  non-JSON response:", raw[:200]); sys.exit()
if "measurements" in d:
    print("\n".join("  " + l for l in d["measurements"].splitlines()))
else:
    print("  error:", d)
'
done
