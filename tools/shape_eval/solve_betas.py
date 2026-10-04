"""Recover SMPL-X betas from exported T-pose GLBs.

With zero pose, the exported vertices are  X = s * (T + S @ beta) + t,
which is linear in (s, s*beta, t). Least squares gives beta exactly;
the residual tells us whether the model assumption holds.
"""
import json, struct, sys
import numpy as np

S_DIR = sys.argv[1] if len(sys.argv) > 1 else '.'
NB = 10
HEIGHT_M = 1.70


def glb_positions(path):
    data = open(path, 'rb').read()
    magic, _, _ = struct.unpack_from('<4sII', data, 0)
    assert magic == b'glTF', f'{path}: not a GLB'
    off, gltf, binc = 12, None, None
    while off < len(data):
        clen, ctype = struct.unpack_from('<II', data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:
            binc = chunk
        off += 8 + clen
    acc = gltf['accessors'][gltf['meshes'][0]['primitives'][0]['attributes']['POSITION']]
    bv = gltf['bufferViews'][acc['bufferView']]
    assert acc['componentType'] == 5126 and acc['type'] == 'VEC3'
    start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    stride = bv.get('byteStride', 12)
    n = acc['count']
    raw = np.frombuffer(binc, dtype=np.uint8, count=stride * (n - 1) + 12, offset=start)
    return np.lib.stride_tricks.as_strided(
        raw.view(np.float32), shape=(n, 3), strides=(stride, 4)).astype(np.float64)


m = np.load(f'{S_DIR}/SMPLX_NEUTRAL.npz', allow_pickle=True)
T = m['v_template'].astype(np.float64)                 # (N,3)
SD = m['shapedirs'][:, :, :NB].astype(np.float64)      # (N,3,NB)
N = T.shape[0]
print(f'model: v_template {T.shape}, shapedirs {m["shapedirs"].shape}')

# design matrix: columns [T | shapedirs_0..9 | ex ey ez]
A = np.zeros((N * 3, 1 + NB + 3))
A[:, 0] = T.reshape(-1)
A[:, 1:1 + NB] = SD.reshape(N * 3, NB)
for k in range(3):
    A[k::3, 1 + NB + k] = 1.0


def recover(path):
    X = glb_positions(path)
    assert X.shape[0] == N, f'{path}: {X.shape[0]} verts, expected {N}'
    sol, *_ = np.linalg.lstsq(A, X.reshape(-1), rcond=None)
    s, b, t = sol[0], sol[1:1 + NB], sol[1 + NB:]
    resid = X.reshape(-1) - A @ sol
    return X, s, b / s, t, np.sqrt(np.mean(resid ** 2)) * 1000


np.set_printoptions(precision=3, suppress=True, linewidth=140)
res = {}
for name in ('female', 'male', 'stock'):
    try:
        X, s, beta, t, rms = recover(f'{S_DIR}/{name}.glb')
    except FileNotFoundError:
        print(f'{name}: missing'); continue
    res[name] = (X, beta)
    h = X[:, 1].max() - X[:, 1].min()
    print(f'\n{name:7s} scale={s:.4f}  height={h:.4f}m  fit residual RMS={rms:.4f} mm')
    print(f'  beta = {beta}')
    print(f'  |beta| = {np.linalg.norm(beta):.3f}   (population: each component ~N(0,1), typical |beta| ~3.2)')

names = list(res)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        d = np.linalg.norm(res[a][0] - res[b][0], axis=1) * 1000
        print(f'{a} vs {b}: vertex RMS {np.sqrt((d**2).mean()):.2f} mm, max {d.max():.2f} mm; '
              f'|dbeta| {np.linalg.norm(res[a][1] - res[b][1]):.3f}')

# How much of each beta component survives normalising to a fixed height?
print(f'\nper-component sensitivity: RMS vertex shift for beta_k = +1, raw vs after height-normalising to {HEIGHT_M} m')


def height_norm(V):
    return V * (HEIGHT_M / (V[:, 1].max() - V[:, 1].min()))


base_n = height_norm(T)
for k in range(NB):
    V = T + SD[:, :, k]
    raw = np.sqrt((np.linalg.norm(V - T, axis=1) ** 2).mean()) * 1000
    nrm = np.sqrt((np.linalg.norm(height_norm(V) - base_n, axis=1) ** 2).mean()) * 1000
    print(f'  beta[{k}]: raw {raw:6.1f} mm   height-normalised {nrm:6.1f} mm   ({100*nrm/raw:4.0f}% survives)')
