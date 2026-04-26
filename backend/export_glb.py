"""
Export SMPL-X body as a skinned GLB with VRM-compatible humanoid bone names.
Frontend (three-vrm + Kalidokit) can drive bones directly via skeleton.getBoneByName().
"""
import json
import struct
import tempfile
import numpy as np
import torch

# SMPL-X joint indices we export: body (0-21) + hands (25-54); skip jaw(22) eyes(23,24)
_SMPLX_COLS = list(range(22)) + list(range(25, 55))  # 52 joints total

# ── Joint names in GLB order (indices into _SMPLX_COLS) ──────────────────────

_JOINT_NAMES = [
    # body (0-21)
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2', 'left_ankle',
    'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    # left hand (22-36) — SMPL-X joints 25-39
    'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3',
    # right hand (37-51) — SMPL-X joints 40-54
    'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3',
]

_PARENTS = [
    # body
    -1,  # 0  pelvis
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    # left hand — all roots parent to left_wrist (20)
    20,  # 22 left_index1
    22,  # 23 left_index2
    23,  # 24 left_index3
    20,  # 25 left_middle1
    25,  # 26 left_middle2
    26,  # 27 left_middle3
    20,  # 28 left_pinky1
    28,  # 29 left_pinky2
    29,  # 30 left_pinky3
    20,  # 31 left_ring1
    31,  # 32 left_ring2
    32,  # 33 left_ring3
    20,  # 34 left_thumb1
    34,  # 35 left_thumb2
    35,  # 36 left_thumb3
    # right hand — all roots parent to right_wrist (21)
    21,  # 37 right_index1
    37,  # 38 right_index2
    38,  # 39 right_index3
    21,  # 40 right_middle1
    40,  # 41 right_middle2
    41,  # 42 right_middle3
    21,  # 43 right_pinky1
    43,  # 44 right_pinky2
    44,  # 45 right_pinky3
    21,  # 46 right_ring1
    46,  # 47 right_ring2
    47,  # 48 right_ring3
    21,  # 49 right_thumb1
    49,  # 50 right_thumb2
    50,  # 51 right_thumb3
]

# VRM 1.0 humanoid bone names
_VRM_NAMES = {
    # body
    'pelvis':         'Hips',
    'left_hip':       'LeftUpperLeg',
    'right_hip':      'RightUpperLeg',
    'spine1':         'Spine',
    'left_knee':      'LeftLowerLeg',
    'right_knee':     'RightLowerLeg',
    'spine2':         'Chest',
    'left_ankle':     'LeftFoot',
    'right_ankle':    'RightFoot',
    'spine3':         'UpperChest',
    'left_foot':      'LeftToes',
    'right_foot':     'RightToes',
    'neck':           'Neck',
    'left_collar':    'LeftShoulder',
    'right_collar':   'RightShoulder',
    'head':           'Head',
    'left_shoulder':  'LeftUpperArm',
    'right_shoulder': 'RightUpperArm',
    'left_elbow':     'LeftLowerArm',
    'right_elbow':    'RightLowerArm',
    'left_wrist':     'LeftHand',
    'right_wrist':    'RightHand',
    # left hand
    'left_index1':    'LeftIndexProximal',
    'left_index2':    'LeftIndexIntermediate',
    'left_index3':    'LeftIndexDistal',
    'left_middle1':   'LeftMiddleProximal',
    'left_middle2':   'LeftMiddleIntermediate',
    'left_middle3':   'LeftMiddleDistal',
    'left_pinky1':    'LeftLittleProximal',
    'left_pinky2':    'LeftLittleIntermediate',
    'left_pinky3':    'LeftLittleDistal',
    'left_ring1':     'LeftRingProximal',
    'left_ring2':     'LeftRingIntermediate',
    'left_ring3':     'LeftRingDistal',
    'left_thumb1':    'LeftThumbMetacarpal',
    'left_thumb2':    'LeftThumbProximal',
    'left_thumb3':    'LeftThumbDistal',
    # right hand
    'right_index1':   'RightIndexProximal',
    'right_index2':   'RightIndexIntermediate',
    'right_index3':   'RightIndexDistal',
    'right_middle1':  'RightMiddleProximal',
    'right_middle2':  'RightMiddleIntermediate',
    'right_middle3':  'RightMiddleDistal',
    'right_pinky1':   'RightLittleProximal',
    'right_pinky2':   'RightLittleIntermediate',
    'right_pinky3':   'RightLittleDistal',
    'right_ring1':    'RightRingProximal',
    'right_ring2':    'RightRingIntermediate',
    'right_ring3':    'RightRingDistal',
    'right_thumb1':   'RightThumbMetacarpal',
    'right_thumb2':   'RightThumbProximal',
    'right_thumb3':   'RightThumbDistal',
}

N_JOINTS = len(_JOINT_NAMES)  # 52

# 10 SMPL-X expression PCA components exported as morph targets.
# Names match what Three.js will populate in mesh.morphTargetDictionary.
EXPRESSION_MORPH_NAMES = [f"expr_{i}" for i in range(10)]
# Strength to drive each basis vector (how far we push the expression).
# 2.0 gives visible but not exaggerated deformation for all 10 components.
_EXPR_STRENGTH = 2.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _top4_weights(lbs_weights, vertices=None, joint_positions=None):
    """
    lbs_weights:     (N, 55) float32 — SMPL-X LBS weights (all 55 joints)
    vertices:        (N, 3)  float32 — T-pose vertex positions
    joint_positions: (52, 3) float32 — T-pose joint positions for exported joints

    We export 52 of 55 joints (skip jaw=22, eyes=23,24). Vertices whose summed weight
    on the 52 exported joints is < 0.3 (jaw/eye verts) are clamped to nearest joint.
    """
    w = np.asarray(lbs_weights[:, _SMPLX_COLS], dtype=np.float32).copy()

    if vertices is not None and joint_positions is not None:
        verts = np.asarray(vertices, dtype=np.float32)
        jpos  = np.asarray(joint_positions, dtype=np.float32)  # (52, 3)
        problem = np.where(w.sum(axis=1) < 0.3)[0]             # jaw/eye verts
        if len(problem):
            dists = np.linalg.norm(
                verts[problem, None, :] - jpos[None, :, :], axis=2
            )                                                   # (M, 52)
            nearest = dists.argmin(axis=1)                      # (M,)
            w[problem] = 0.0
            w[problem, nearest] = 1.0

    top4 = np.argsort(w, axis=1)[:, -4:][:, ::-1]              # (N, 4) descending
    gathered = np.take_along_axis(w, top4, axis=1)
    row_sum = gathered.sum(axis=1, keepdims=True)
    gathered /= np.where(row_sum < 1e-6, 1.0, row_sum)         # renormalize
    return top4.astype(np.uint16), gathered.astype(np.float32)


def _col_major_inv_bind(joint_pos):
    """
    Returns flat (16,) float32 column-major inverse-bind matrix for a joint at
    world position joint_pos in a T-pose (rotation = identity).
    """
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = -joint_pos          # inverse translation, row-major
    return m.T.ravel()             # GLTF wants column-major


def _aligned(data: bytes) -> bytes:
    pad = (4 - len(data) % 4) % 4
    return data + b'\x00' * pad


def _pack(*arrays):
    """Pack numpy arrays sequentially with 4-byte alignment. Returns (blob, offsets)."""
    blob, offsets, off = [], [], 0
    for arr in arrays:
        b = arr.tobytes()
        offsets.append(off)
        aligned = _aligned(b)
        blob.append(aligned)
        off += len(aligned)
    return b''.join(blob), offsets


# ── expression morphs ────────────────────────────────────────────────────────

def compute_expression_morphs(smplx_model, betas, scale_factor: float = 1.0) -> np.ndarray:
    """
    Return (10, N, 3) float32 scaled vertex deltas — one per SMPL-X expression component.

    Each delta[i] = vertices(expression[i]=_EXPR_STRENGTH) - vertices(expression=0),
    scaled by scale_factor so they match the height-scaled mesh in the GLB.
    """
    n = len(EXPRESSION_MORPH_NAMES)
    zero_expr = torch.zeros(1, n)
    kw = dict(betas=betas, body_pose=torch.zeros(1, 63),
              global_orient=torch.zeros(1, 3), return_verts=True)

    with torch.no_grad():
        base_v = smplx_model(expression=zero_expr, **kw).vertices[0].numpy()
        deltas = np.zeros((n, len(base_v), 3), dtype=np.float32)
        for i in range(n):
            expr = zero_expr.clone()
            expr[0, i] = _EXPR_STRENGTH
            v_i = smplx_model(expression=expr, **kw).vertices[0].numpy()
            deltas[i] = (v_i - base_v).astype(np.float32) * scale_factor

    return deltas


# ── main export ───────────────────────────────────────────────────────────────

def export_skinned_glb(vertices, faces, joints_world, lbs_weights,
                       morph_deltas: np.ndarray | None = None) -> str:
    """
    Build a skinned GLB with VRM humanoid bone names and optional morph targets.

    vertices:     (N, 3)   float32 — scaled T-pose vertices
    faces:        (F, 3)   int32/uint32 — triangle indices
    joints_world: (>=55,3) float32 — scaled joint world positions (all 55 SMPL-X joints)
    lbs_weights:  (N, 55)  float32 — SMPL-X per-vertex LBS weights
    morph_deltas: (M, N, 3) float32 — optional M morph target vertex deltas (scaled)

    Returns path to a temp .glb file (caller owns cleanup).
    """
    verts  = np.asarray(vertices,                         dtype=np.float32)
    tris   = np.asarray(faces,                            dtype=np.uint32)
    jpos   = np.asarray(joints_world, dtype=np.float32)[_SMPLX_COLS]  # (52, 3)
    lbsw   = np.asarray(lbs_weights,                      dtype=np.float32)

    N, F = len(verts), len(tris)

    # Per-vertex normals
    fn = np.cross(verts[tris[:,1]] - verts[tris[:,0]],
                  verts[tris[:,2]] - verts[tris[:,0]])
    normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(normals, tris[:, i], fn)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(nlen < 1e-8, 1.0, nlen)
    normals = normals.astype(np.float32)

    j_idx, j_wt = _top4_weights(lbsw, verts, jpos)
    inv_binds = np.stack([_col_major_inv_bind(jpos[j]) for j in range(N_JOINTS)])

    # Morph target arrays
    n_morphs = 0
    morph_arrays: list[np.ndarray] = []
    if morph_deltas is not None:
        n_morphs = len(morph_deltas)
        morph_arrays = [np.asarray(d, dtype=np.float32) for d in morph_deltas]

    bin_blob, offsets = _pack(verts, normals, tris, j_idx, j_wt, inv_binds, *morph_arrays)

    ARRAY_BUF, ELEM_BUF = 34962, 34963
    bvs = [
        {"buffer": 0, "byteOffset": offsets[0], "byteLength": N * 12,        "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[1], "byteLength": N * 12,        "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[2], "byteLength": F * 3 * 4,     "target": ELEM_BUF},
        {"buffer": 0, "byteOffset": offsets[3], "byteLength": N * 4 * 2,     "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[4], "byteLength": N * 4 * 4,     "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[5], "byteLength": N_JOINTS * 64             },
    ]
    for i in range(n_morphs):
        bvs.append({"buffer": 0, "byteOffset": offsets[6 + i],
                    "byteLength": N * 12, "target": ARRAY_BUF})

    FLOAT, UINT32, UINT16 = 5126, 5125, 5123
    accs = [
        {"bufferView": 0, "componentType": FLOAT,  "count": N,         "type": "VEC3",
         "min": verts.min(0).tolist(), "max": verts.max(0).tolist()},
        {"bufferView": 1, "componentType": FLOAT,  "count": N,         "type": "VEC3"},
        {"bufferView": 2, "componentType": UINT32, "count": F * 3,     "type": "SCALAR"},
        {"bufferView": 3, "componentType": UINT16, "count": N,         "type": "VEC4"},
        {"bufferView": 4, "componentType": FLOAT,  "count": N,         "type": "VEC4"},
        {"bufferView": 5, "componentType": FLOAT,  "count": N_JOINTS,  "type": "MAT4"},
    ]
    for i, d in enumerate(morph_arrays):
        accs.append({"bufferView": 6 + i, "componentType": FLOAT, "count": N, "type": "VEC3",
                     "min": d.min(0).tolist(), "max": d.max(0).tolist()})

    # Mesh primitive
    primitive = {
        "attributes": {"POSITION": 0, "NORMAL": 1, "JOINTS_0": 3, "WEIGHTS_0": 4},
        "indices": 2, "mode": 4,
    }
    if n_morphs:
        primitive["targets"] = [{"POSITION": 6 + i} for i in range(n_morphs)]

    mesh_def: dict = {"name": "Body", "primitives": [primitive]}
    if n_morphs:
        mesh_def["weights"] = [0.0] * n_morphs
        mesh_def["extras"]  = {"targetNames": EXPRESSION_MORPH_NAMES[:n_morphs]}

    # Node layout: 0=Armature root, 1..N_JOINTS=joints, N_JOINTS+1=mesh node
    J0 = 1
    MESH_NODE = N_JOINTS + 1

    nodes = [{"name": "Armature", "children": [J0, MESH_NODE]}]
    for j in range(N_JOINTS):
        p = _PARENTS[j]
        local = (jpos[j] - jpos[p]).tolist() if p >= 0 else jpos[j].tolist()
        children = [i + J0 for i, par in enumerate(_PARENTS) if par == j]
        node = {"name": _VRM_NAMES[_JOINT_NAMES[j]], "translation": local}
        if children:
            node["children"] = children
        nodes.append(node)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})

    gltf = {
        "asset": {"version": "2.0", "generator": "smplx-measure"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [mesh_def],
        "skins": [{"name": "Armature",
                   "skeleton": J0,
                   "joints": list(range(J0, J0 + N_JOINTS)),
                   "inverseBindMatrices": 5}],
        "accessors": accs,
        "bufferViews": bvs,
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    # Serialize to GLB
    json_bytes = json.dumps(gltf, separators=(',', ':')).encode()
    json_bytes += b' ' * ((4 - len(json_bytes) % 4) % 4)  # pad to 4-byte boundary

    JSON_TYPE = 0x4E4F534A
    BIN_TYPE  = 0x004E4942
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    header     = struct.pack('<4sII', b'glTF', 2, total)
    json_chunk = struct.pack('<II', len(json_bytes), JSON_TYPE) + json_bytes
    bin_chunk  = struct.pack('<II', len(bin_blob),  BIN_TYPE)  + bin_blob

    tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    tmp.write(header + json_chunk + bin_chunk)
    tmp.close()
    return tmp.name
