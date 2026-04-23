#!/bin/bash
# HF Spaces startup script — clones PyMAF-X and fetches data files.
# Checkpoint must be stored as a HF Dataset private file and pulled via token.
set -e

PYMAFX_DIR="/tmp/PyMAF-X"

# ── clone PyMAF-X if not present ─────────────────────────────────────────────
if [ ! -d "$PYMAFX_DIR" ]; then
  echo "Cloning PyMAF-X..."
  git clone --depth 1 https://github.com/HongwenZhang/PyMAF-X.git "$PYMAFX_DIR"
fi

# ── patch numpy 2.x incompatibility ─────────────────────────────────────────
sed -i 's/from numpy.lib.twodim_base import triu_indices_from/from numpy import triu_indices_from/' \
  "$PYMAFX_DIR/models/maf_extractor.py"

# ── fetch base data files (smpl downsampling, mean params, etc.) ─────────────
cd "$PYMAFX_DIR"
mkdir -p data/pretrained_model data/partial_mesh

# smpl_downsampling.npz
if [ ! -f data/smpl_downsampling.npz ]; then
  wget -q https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz \
    -O data/smpl_downsampling.npz
fi

# mano_downsampling.npz
if [ ! -f data/mano_downsampling.npz ]; then
  wget -q https://github.com/microsoft/MeshGraphormer/raw/main/src/modeling/data/mano_downsampling.npz \
    -O data/mano_downsampling.npz
fi

# ── download checkpoint from HF Hub (stored as private dataset) ───────────────
# Requires HF_TOKEN secret set in Space settings
# Dataset: kaustubh/smplx-measure-assets  (private, checkpoint + partial_mesh + SMPL-X models)
python - <<'PYEOF'
import os, sys
token = os.environ.get('HF_TOKEN', '')
if not token:
    print("WARNING: HF_TOKEN not set — checkpoint download will fail")
    sys.exit(0)

from huggingface_hub import hf_hub_download
import shutil, pathlib

REPO = 'kaustubh/smplx-measure-assets'
DEST = '/tmp/PyMAF-X/data'

files = [
    ('pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt', 'pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt'),
    ('partial_mesh/smplx_arm_vids.npz',   'partial_mesh/smplx_arm_vids.npz'),
    ('partial_mesh/smplx_face_vids.npz',  'partial_mesh/smplx_face_vids.npz'),
    ('partial_mesh/smplx_larm_vids.npz',  'partial_mesh/smplx_larm_vids.npz'),
    ('partial_mesh/smplx_rarm_vids.npz',  'partial_mesh/smplx_rarm_vids.npz'),
    ('partial_mesh/smplx_lhand_vids.npz', 'partial_mesh/smplx_lhand_vids.npz'),
    ('partial_mesh/smplx_rhand_vids.npz', 'partial_mesh/smplx_rhand_vids.npz'),
    ('partial_mesh/smplx_lwrist_vids.npz','partial_mesh/smplx_lwrist_vids.npz'),
    ('partial_mesh/smplx_rwrist_vids.npz','partial_mesh/smplx_rwrist_vids.npz'),
    ('partial_mesh/smplx_forearm_vids.npz','partial_mesh/smplx_forearm_vids.npz'),
]

for repo_path, local_rel in files:
    local_path = os.path.join(DEST, local_rel)
    if os.path.exists(local_path):
        continue
    print(f'Downloading {repo_path}...')
    src = hf_hub_download(repo_id=REPO, filename=repo_path,
                          repo_type='dataset', token=token)
    pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, local_path)

print('PyMAF-X data ready.')
PYEOF

echo "setup.sh complete"
