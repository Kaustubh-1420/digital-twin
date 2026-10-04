#!/bin/bash
# macOS port of setup.sh: clone PyMAF-X + fetch all assets into /tmp, using the cached HF token.
set -e
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
D=/tmp/PyMAF-X

[ -d $D ] || git clone -q --depth 1 https://github.com/HongwenZhang/PyMAF-X.git $D
sed -i '' 's/from numpy.lib.twodim_base import triu_indices_from/from numpy import triu_indices_from/' $D/models/maf_extractor.py
mkdir -p $D/data/pretrained_model $D/data/partial_mesh
[ -f $D/data/smpl_downsampling.npz ] || curl -sSL -o $D/data/smpl_downsampling.npz https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz
[ -f $D/data/mano_downsampling.npz ] || curl -sSL -o $D/data/mano_downsampling.npz https://github.com/microsoft/MeshGraphormer/raw/main/src/modeling/data/mano_downsampling.npz
echo "PyMAF-X cloned + patched"

# Reuse setup.sh's own Python download block verbatim, with HF_TOKEN from the local cache.
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
sed -n "/^python - <<'PYEOF'/,/^PYEOF/p" $REPO_DIR/setup.sh | sed '1d;$d' | $REPO_DIR/.venv/bin/python -

ls -la $D/data/pretrained_model/ /tmp/smplx-models/smplx/
