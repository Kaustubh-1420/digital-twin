"""
Anthropometric measurements from SMPL-X T-pose output.

Circumferences (chest/waist/hip): horizontal plane slice on mesh at
joint-derived Y heights → perimeter of intersection polygon.

Lengths (inseam, arm): sum of Euclidean distances along the joint chain.
Shoulder width: Euclidean distance between shoulder joints.

All inputs use the raw smplx model output (vertices, faces, joints) —
NOT a round-tripped .obj file, which reorders vertices.

SMPL-X body joint indices (0-21):
  0=pelvis, 1=left_hip, 2=right_hip, 3=spine1,
  4=left_knee, 5=right_knee, 6=spine2, 7=left_ankle,
  8=right_ankle, 9=spine3, 10=left_foot, 11=right_foot,
  12=neck, 13=left_collar, 14=right_collar, 15=head,
  16=left_shoulder, 17=right_shoulder, 18=left_elbow,
  19=right_elbow, 20=left_wrist, 21=right_wrist
"""

import numpy as np
import trimesh


def _plane_slice_perimeter(vertices: np.ndarray, faces: np.ndarray, y: float) -> float:
    """Intersect mesh with horizontal plane at height y, return full perimeter.
    Used for waist and hip where arms don't intersect the slice plane.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    section = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if section is None:
        return 0.0
    path2d, _ = section.to_planar()
    if len(path2d.entities) == 0:
        return 0.0

    return max(e.length(path2d.vertices) for e in path2d.entities)


def _torso_ellipse_circumference(vertices: np.ndarray, y: float,
                                  x_limit: float, band: float = 0.025) -> float:
    """
    Estimate torso circumference at height y via ellipse approximation.
    Selects only vertices within x_limit of center (excludes arms).
    Uses Ramanujan's ellipse perimeter approximation.
    Accurate to ~1% for typical chest/torso cross-sections.
    """
    verts_centered = vertices.copy()
    verts_centered[:, 0] -= vertices[:, 0].mean()
    mask = (np.abs(verts_centered[:, 1] - y) < band) & (np.abs(verts_centered[:, 0]) < x_limit)
    verts = verts_centered[mask]
    if len(verts) < 6:
        return 0.0
    a = (verts[:, 0].max() - verts[:, 0].min()) / 2  # half-width (X)
    b = (verts[:, 2].max() - verts[:, 2].min()) / 2  # half-depth (Z)
    # Ramanujan approximation
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def extract_measurements(vertices: np.ndarray, faces: np.ndarray,
                         joints: np.ndarray) -> dict:
    """
    Extract 6 anthropometric measurements from a T-pose SMPL-X output.

    Args:
        vertices: (10475, 3) float, Y-up, metres — raw smplx output
        faces:    (20908, 3) int
        joints:   (127, 3) float — raw smplx joints output, body joints at 0-21

    Returns:
        dict of measurement name → value in centimetres
    """
    J = joints  # (127, 3), use indices 0-21 for body

    # --- Plane heights from joints ---
    # Chest: at collar height — arm-free ellipse estimate (arms at same Y, can't plane-slice)
    chest_y   = (J[13, 1] + J[14, 1]) / 2 - 0.02
    # Waist: midpoint between spine2 (6) and pelvis (0)
    waist_y   = (J[6, 1] + J[0, 1]) / 2
    # Hip: at hip joint level (1, 2)
    hip_y     = (J[1, 1] + J[2, 1]) / 2

    # Torso half-width: shoulder joint X extent (arms start here, exclude beyond this)
    torso_x_limit = (abs(J[16, 0]) + abs(J[17, 0])) / 2 + 0.01

    # Chest via ellipse (arm-proof); waist+hip via plane slice (below arm level)
    chest_circ  = _torso_ellipse_circumference(vertices, chest_y, torso_x_limit)
    waist_circ  = _plane_slice_perimeter(vertices, faces, waist_y)
    hip_circ    = _plane_slice_perimeter(vertices, faces, hip_y)

    # --- Lengths along joint chains ---
    # Inseam: pelvis(0) → left_hip(1) → left_knee(4) → left_ankle(7)
    inseam = (np.linalg.norm(J[0] - J[1]) +
              np.linalg.norm(J[1] - J[4]) +
              np.linalg.norm(J[4] - J[7]))

    # Arm length: left_shoulder(16) → left_elbow(18) → left_wrist(20)
    arm_len = (np.linalg.norm(J[16] - J[18]) +
               np.linalg.norm(J[18] - J[20]))

    # --- Shoulder width: straight line between shoulder joints ---
    shoulder_w = float(np.linalg.norm(J[16] - J[17]))

    def m_to_cm(v):
        return round(float(v) * 100, 1)

    return {
        'chest_cm':      m_to_cm(chest_circ),
        'waist_cm':      m_to_cm(waist_circ),
        'hip_cm':        m_to_cm(hip_circ),
        'shoulder_cm':   m_to_cm(shoulder_w),
        'inseam_cm':     m_to_cm(inseam),
        'arm_length_cm': m_to_cm(arm_len),
    }
