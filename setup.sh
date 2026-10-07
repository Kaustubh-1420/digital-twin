#!/bin/bash
# HF Spaces startup script — fetches the SMPL-X model and NLF weights into /tmp.
set -e

# ── NLF weights (public GitHub release; noncommercial research use) ──────────
NLF=/tmp/nlf/nlf_l_multi_0.3.2.torchscript
if [ ! -f "$NLF" ]; then
  echo "Downloading NLF weights..."
  mkdir -p /tmp/nlf
  wget -q https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript -O "$NLF.part"
  mv "$NLF.part" "$NLF"
fi

# ── SMPL-X neutral model (private HF dataset; needs the HF_TOKEN secret) ─────
python - <<'PYEOF'
import os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download

dest = '/tmp/smplx-models/smplx/SMPLX_NEUTRAL.npz'
if not os.path.exists(dest):
    token = os.environ.get('HF_TOKEN', '')
    if not token:
        sys.exit('HF_TOKEN not set: cannot download the SMPL-X model')
    print('Downloading smplx/SMPLX_NEUTRAL.npz...')
    src = hf_hub_download(repo_id='Kaustubh1420/smplx-measure-assets', filename='smplx/SMPLX_NEUTRAL.npz',
                          repo_type='dataset', token=token)
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
print('SMPL-X model ready.')
PYEOF

echo "setup.sh complete"
