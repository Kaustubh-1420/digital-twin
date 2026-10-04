#!/bin/bash
# Fetch SMPL-X model + live GLBs for both samples + stock mannequin, for beta recovery.
set -u
S="$(cd "$(dirname "$0")" && pwd)/stress/live"; mkdir -p "$S"
P="$(cd "$(dirname "$0")/../.." && pwd)/frontend/public/samples"
SITE=https://digital-twin-ai.vercel.app
cd "$S"
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE

curl -sS -L -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" -o SMPLX_NEUTRAL.npz \
  -w "npz: %{http_code} %{size_download}B\n" \
  https://huggingface.co/datasets/Kaustubh1420/smplx-measure-assets/resolve/main/smplx/SMPLX_NEUTRAL.npz

curl -sS -o stock.glb -w "stock: %{http_code} %{size_download}B\n" "$SITE/sample-avatar.glb"

for g in female male; do
  curl -sS -m 120 -F "image=@$P/sample-$g.jpg;type=image/jpeg" -F heightCm=170 -o "pred_$g.json" \
    -w "predict $g: %{http_code} %{time_total}s\n" "$SITE/api/predict"
  url=$("$(dirname "$S")/../../../.venv/bin/python" -c "import json,sys,urllib.parse;print(urllib.parse.quote(json.load(open('pred_$g.json'))['glbUrl'],safe=''))") || { head -c 300 "pred_$g.json"; echo; continue; }
  curl -sS -m 120 -o "$g.glb" -w "glb $g: %{http_code} %{size_download}B %{time_total}s\n" "$SITE/api/glb?url=$url"
done
