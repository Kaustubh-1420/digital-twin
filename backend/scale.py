"""
Height-based mesh scaling.

SMPL-X with estimated betas gives correct relative proportions but
ambiguous absolute scale from a single photo. User-provided height
anchors the scale so all measurements are in real-world centimetres.
"""

import numpy as np


def scale_to_height(vertices: np.ndarray, height_cm: float) -> np.ndarray:
    """
    Uniformly scale mesh vertices so the mesh height matches height_cm.

    Mesh height is defined as max(Y) - min(Y) — top of head to bottom of feet.
    All downstream measurements inherit the correct scale automatically.

    Args:
        vertices:   (N, 3) float array, Y-up
        height_cm:  user-provided standing height in centimetres

    Returns:
        scaled vertices (N, 3), same dtype
    """
    mesh_height_m = float(vertices[:, 1].max() - vertices[:, 1].min())
    if mesh_height_m < 1e-6:
        raise ValueError("Degenerate mesh: height is near zero")

    scale = (height_cm / 100.0) / mesh_height_m
    return vertices * scale
