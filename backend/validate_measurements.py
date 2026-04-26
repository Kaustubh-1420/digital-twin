"""
Task 28: Validate measurement accuracy on neutral SMPL-X T-pose.

Loads SMPLX_NEUTRAL with zero betas/poses, scales to 170cm, runs
extract_measurements(), and checks against WHO/ANSUR reference ranges.
"""

import sys
import torch
import smplx
import numpy as np

sys.path.insert(0, ".")
from backend.measurements import extract_measurements
from backend.scale import scale_to_height

MODEL_PATH = "models"
TARGET_HEIGHT_CM = 170.0

# Reference ranges (cm) for a neutral ~170cm adult.
# Sources: ANSUR II, WHO adult anthropometrics, typical SMPLify literature.
RANGES = {
    "chest_cm":      (85.0, 105.0),
    "waist_cm":      (68.0,  92.0),
    "hip_cm":        (85.0, 105.0),
    "shoulder_cm":   (34.0,  50.0),  # acromion-to-acromion, vertex-based
    "inseam_cm":     (68.0,  86.0),  # hip→knee→ankle (no pelvis segment)
    "arm_length_cm": (46.0,  60.0),  # glenohumeral joint chain to wrist
}


def main():
    model = smplx.create(
        MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
    )

    with torch.no_grad():
        output = model(
            betas=torch.zeros(1, 10),
            body_pose=torch.zeros(1, 63),
            global_orient=torch.zeros(1, 3),
            return_verts=True,
        )

    verts  = output.vertices[0].numpy()   # (10475, 3)
    joints = output.joints[0].numpy()     # (127, 3)
    faces  = model.faces                  # (20908, 3)

    verts_s, joints_s = scale_to_height(verts, TARGET_HEIGHT_CM, joints)

    actual_height = (verts_s[:, 1].max() - verts_s[:, 1].min()) * 100
    print(f"\nMesh height after scaling: {actual_height:.1f} cm  (target {TARGET_HEIGHT_CM} cm)")

    m = extract_measurements(verts_s, faces, joints_s)

    print(f"\n{'Measurement':<18} {'Value':>8}   {'Range':>16}   {'Pass?':>6}")
    print("-" * 58)

    all_pass = True
    for key, (lo, hi) in RANGES.items():
        val = m[key]
        ok  = lo <= val <= hi
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"{key:<18} {val:>7.1f}   [{lo:.0f} – {hi:.0f}]{' ':>6}   {flag:>6}")

    print()
    if all_pass:
        print("ALL PASS — measurement pipeline looks correct.")
    else:
        print("SOME CHECKS FAILED — review values above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
