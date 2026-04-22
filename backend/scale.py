"""
Height-based mesh scaling.

SMPL-X with estimated betas gives correct relative proportions but
ambiguous absolute scale from a single photo. User-provided height
anchors the scale so all measurements are in real-world centimetres.
"""

import numpy as np


def scale_to_height(
    vertices: np.ndarray,
    height_cm: float,
    joints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Uniformly scale mesh so its height matches height_cm.

    Args:
        vertices:   (N, 3) float array, Y-up
        height_cm:  user-provided standing height in centimetres
        joints:     (J, 3) float array — scaled by the same factor if provided

    Returns:
        (scaled_vertices, scaled_joints) — scaled_joints is None if joints was None
    """
    mesh_height_m = float(vertices[:, 1].max() - vertices[:, 1].min())
    if mesh_height_m < 1e-6:
        raise ValueError("Degenerate mesh: height is near zero")

    scale = (height_cm / 100.0) / mesh_height_m
    scaled_joints = joints * scale if joints is not None else None
    return vertices * scale, scaled_joints
