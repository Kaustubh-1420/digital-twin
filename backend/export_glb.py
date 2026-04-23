"""
Export SMPL-X body as a skinned GLB with VRM-compatible humanoid bone names.
Frontend (three-vrm + Kalidokit) can drive bones directly via skeleton.getBoneByName().
"""
import json
import struct
import tempfile
import numpy as np

# ── SMPL-X body joint topology (first 22 joints) ─────────────────────────────

_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2', 'left_ankle',
    'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
]

_PARENTS = [
    -1,  # pelvis
     0,  # left_hip
     0,  # right_hip
     0,  # spine1
     1,  # left_knee
     2,  # right_knee
     3,  # spine2
     4,  # left_ankle
     5,  # right_ankle
     6,  # spine3
     7,  # left_foot
     8,  # right_foot
     9,  # neck
     9,  # left_collar
     9,  # right_collar
    12,  # head
    13,  # left_shoulder
    14,  # right_shoulder
    16,  # left_elbow
    17,  # right_elbow
    18,  # left_wrist
    19,  # right_wrist
]

# VRM 1.0 humanoid bone names (Kalidokit PoseSolver uses these exact strings)
_VRM_NAMES = {
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
}

N_JOINTS = len(_JOINT_NAMES)  # 22


# ── helpers ───────────────────────────────────────────────────────────────────

def _top4_weights(lbs_weights):
    """
    lbs_weights: (N, >=22) float32 — SMPL-X LBS weights (first 22 cols = body joints)
    Returns joint_indices (N,4) uint16, skin_weights (N,4) float32 — top-4 per vertex.
    """
    w = np.asarray(lbs_weights[:, :N_JOINTS], dtype=np.float32)
    top4 = np.argsort(w, axis=1)[:, -4:][:, ::-1]          # (N, 4) descending
    gathered = np.take_along_axis(w, top4, axis=1)
    row_sum = gathered.sum(axis=1, keepdims=True)
    gathered /= np.where(row_sum < 1e-6, 1.0, row_sum)     # renormalize
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


# ── main export ───────────────────────────────────────────────────────────────

def export_skinned_glb(vertices, faces, joints_world, lbs_weights) -> str:
    """
    Build a skinned GLB with VRM humanoid bone names.

    vertices:     (N, 3)   float32 — scaled T-pose vertices
    faces:        (F, 3)   int32/uint32 — triangle indices
    joints_world: (>=22,3) float32 — scaled joint world positions
    lbs_weights:  (N, >=22) float32 — SMPL-X per-vertex LBS weights

    Returns path to a temp .glb file (caller owns cleanup).
    """
    verts  = np.asarray(vertices,          dtype=np.float32)
    tris   = np.asarray(faces,             dtype=np.uint32)
    jpos   = np.asarray(joints_world[:N_JOINTS], dtype=np.float32)
    lbsw   = np.asarray(lbs_weights,       dtype=np.float32)

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

    j_idx, j_wt = _top4_weights(lbsw)
    inv_binds = np.stack([_col_major_inv_bind(jpos[j]) for j in range(N_JOINTS)])

    bin_blob, offsets = _pack(verts, normals, tris, j_idx, j_wt, inv_binds)

    # Buffer views (one per data block)
    ARRAY_BUF, ELEM_BUF = 34962, 34963
    bvs = [
        {"buffer": 0, "byteOffset": offsets[0], "byteLength": N * 12,        "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[1], "byteLength": N * 12,        "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[2], "byteLength": F * 3 * 4,     "target": ELEM_BUF},
        {"buffer": 0, "byteOffset": offsets[3], "byteLength": N * 4 * 2,     "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[4], "byteLength": N * 4 * 4,     "target": ARRAY_BUF},
        {"buffer": 0, "byteOffset": offsets[5], "byteLength": N_JOINTS * 64              },
    ]

    FLOAT, UINT32, UINT16 = 5126, 5125, 5123
    accs = [
        {"bufferView": 0, "componentType": FLOAT,  "count": N,         "type": "VEC3",
         "min": verts.min(0).tolist(), "max": verts.max(0).tolist()},  # POSITION
        {"bufferView": 1, "componentType": FLOAT,  "count": N,         "type": "VEC3"},  # NORMAL
        {"bufferView": 2, "componentType": UINT32, "count": F * 3,     "type": "SCALAR"},  # indices
        {"bufferView": 3, "componentType": UINT16, "count": N,         "type": "VEC4"},  # JOINTS_0
        {"bufferView": 4, "componentType": FLOAT,  "count": N,         "type": "VEC4"},  # WEIGHTS_0
        {"bufferView": 5, "componentType": FLOAT,  "count": N_JOINTS,  "type": "MAT4"},  # invBindMat
    ]

    # Node layout: 0=Armature root, 1..22=joints, 23=mesh node
    J0 = 1  # joint node index offset
    MESH_NODE = N_JOINTS + 1

    nodes = [{
        "name": "Armature",
        "children": [J0, MESH_NODE],
    }]
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
        "meshes": [{"name": "Body", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "JOINTS_0": 3, "WEIGHTS_0": 4},
            "indices": 2, "mode": 4,
        }]}],
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
