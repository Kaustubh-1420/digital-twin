"""
End-to-end test: image → HMR2 → SMPL-X mesh → .obj
Run from project root: python backend/test_inference.py
"""
import sys
from unittest.mock import MagicMock
sys.modules['pyrender'] = MagicMock()

import torch
import numpy as np
import cv2
import smplx
import trimesh
from torch.utils.data import DataLoader
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset

SMPLX_MODEL_PATH = 'models'
TEST_IMAGE = '/tmp/4D-Humans/example_data/images/pexels-anete-lusina-4793258.jpg'
OUT_OBJ = '/tmp/smplx_tpose.obj'


def hmr2_infer(model, model_cfg, img_bgr):
    H, W = img_bgr.shape[:2]
    bbox = np.array([[0, 0, W, H]], dtype=np.float32)
    dataset = ViTDetDataset(model_cfg, img_bgr, bbox)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    with torch.no_grad():
        return model(batch)


def rotmat_to_aa(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices [..., 3, 3] → axis-angle [..., 3] via scipy."""
    from scipy.spatial.transform import Rotation
    shape = rotmat.shape[:-2]
    flat = rotmat.detach().reshape(-1, 3, 3).numpy()
    aa = Rotation.from_matrix(flat).as_rotvec()
    return torch.from_numpy(aa.reshape(*shape, 3)).float()


def smpl_to_smplx_params(hmr2_out):
    """Map HMR2 SMPL output to SMPL-X body params.
    SMPL has 23 body joints; SMPL-X body uses first 21 (hands handled separately).
    Shape betas share the same PCA space — pass through directly.
    Rotation matrices converted to axis-angle for smplx compatibility.
    """
    betas = hmr2_out['pred_smpl_params']['betas']                          # [B, 10]
    body_pose_rotmat = hmr2_out['pred_smpl_params']['body_pose'][:, :21]   # [B, 21, 3, 3]
    global_orient_rotmat = hmr2_out['pred_smpl_params']['global_orient'].squeeze(1)  # [B, 3, 3]

    body_pose_aa = rotmat_to_aa(body_pose_rotmat).reshape(-1, 63)          # [B, 63]
    global_orient_aa = rotmat_to_aa(global_orient_rotmat)                  # [B, 3]

    return {'betas': betas, 'body_pose': body_pose_aa, 'global_orient': global_orient_aa}


def build_smplx_mesh(params, model_path):
    smplx_model = smplx.create(
        model_path=model_path,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        num_betas=10,
        batch_size=1,
    )
    out = smplx_model(
        betas=params['betas'],
        body_pose=params['body_pose'],
        global_orient=params['global_orient'],
        return_verts=True,
    )
    vertices = out.vertices[0].detach().numpy()
    faces = smplx_model.faces
    return vertices, faces, smplx_model


def normalize_to_tpose(betas, smplx_model):
    """Zero out body pose — keep only shape betas.
    Returns vertices, faces, and joints for measurement extraction.
    """
    B = betas.shape[0]
    out = smplx_model(
        betas=betas,
        body_pose=torch.zeros(B, 63),
        global_orient=torch.zeros(B, 3),
        return_verts=True,
    )
    vertices = out.vertices[0].detach().numpy()
    joints   = out.joints[0].detach().numpy()
    return vertices, smplx_model.faces, joints


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=TEST_IMAGE)
    parser.add_argument('--height', type=float, default=175.0,
                        help='User height in cm for scale calibration')
    parser.add_argument('--backend', choices=['hmr2', 'pymafx'], default='pymafx',
                        help='Body fitting backend (pymafx = native SMPL-X, hmr2 = approx betas)')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.image}')

    smplx_model = smplx.create(
        model_path=SMPLX_MODEL_PATH, model_type='smplx',
        gender='neutral', use_pca=False, num_betas=10, batch_size=1,
    )

    if args.backend == 'pymafx':
        print('Loading PyMAF-X backend...')
        sys.path.insert(0, 'backend')
        from pymafx_backend import infer as pymafx_infer
        result = pymafx_infer(img)
        betas_np = result['betas']
        betas = torch.from_numpy(betas_np).unsqueeze(0).float()
    else:
        print('Loading HMR2...')
        model_hmr2, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        model_hmr2 = model_hmr2.eval()
        print('Running HMR2 inference...')
        hmr2_out = hmr2_infer(model_hmr2, model_cfg, img)
        print('Converting SMPL → SMPL-X...')
        params = smpl_to_smplx_params(hmr2_out)
        betas = params['betas']

    print('Normalizing to T-pose...')
    vertices, faces, joints = normalize_to_tpose(betas, smplx_model)

    print(f'Scaling to {args.height:.0f} cm...')
    from scale import scale_to_height
    vertices, joints = scale_to_height(vertices, args.height, joints)

    height_m = vertices[:, 1].max() - vertices[:, 1].min()
    print(f'Vertices: {vertices.shape}, Faces: {faces.shape}, Joints: {joints.shape}')
    print(f'Scaled height: {height_m:.3f} m ({height_m * 100:.1f} cm)')

    print('\nExtracting measurements...')
    from measurements import extract_measurements
    results = extract_measurements(vertices, faces, joints)
    for k, v in results.items():
        print(f'  {k}: {v} cm')

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(OUT_OBJ)
    print(f'\nSaved → {OUT_OBJ}')
