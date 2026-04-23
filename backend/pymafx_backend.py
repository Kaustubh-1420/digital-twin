"""
PyMAF-X backend — native SMPL-X betas from a single photo.
Must be run with CWD = project root (smplx-measure/).

Requires /tmp/PyMAF-X (cloned repo) and checkpoint at:
  /tmp/PyMAF-X/data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt

MediaPipe model cached at /tmp/pose_landmarker_heavy.task (auto-downloaded).
"""
import sys
import os
from unittest.mock import MagicMock

PYMAFX_DIR = '/tmp/PyMAF-X'
MP_MODEL_PATH = '/tmp/pose_landmarker_heavy.task'
CHECKPOINT = os.path.join(PYMAFX_DIR, 'data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt')
CFG = os.path.join(PYMAFX_DIR, 'configs/pymafx_config.yaml')

# MediaPipe 33-landmark → COCO-17 index map
_MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
N_BODY_KP = 17

_model = None  # cached across calls


def _ensure_mocks():
    for mod in ['pyrender', 'neural_renderer', 'openpifpaf']:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def _ensure_pymafx_on_path():
    if PYMAFX_DIR not in sys.path:
        sys.path.insert(0, PYMAFX_DIR)


def _ensure_mp_model():
    if not os.path.exists(MP_MODEL_PATH):
        import ssl, certifi, urllib.request
        ctx = ssl.create_default_context(cafile=certifi.where())
        url = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
               'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task')
        print(f'Downloading MediaPipe model to {MP_MODEL_PATH}...')
        with urllib.request.urlopen(url, context=ctx) as r, open(MP_MODEL_PATH, 'wb') as f:
            f.write(r.read())


def load_model(device=None):
    global _model
    if _model is not None:
        return _model

    _ensure_mocks()
    _ensure_pymafx_on_path()

    import torch
    from core.cfgs import update_cfg
    from core import path_config
    from models import pymaf_net

    orig_cwd = os.getcwd()
    os.chdir(PYMAFX_DIR)
    try:
        update_cfg(CFG)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = pymaf_net(path_config.SMPL_MEAN_PARAMS, is_train=False).to(device)
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=True)
        model.eval()
        _model = (model, device)
        print(f'PyMAF-X loaded on {device}')
    finally:
        os.chdir(orig_cwd)

    return _model


def detect_body_keypoints(img_rgb):
    """Returns (17, 3) keypoints [x, y, visibility] in pixel coords."""
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    from mediapipe.tasks.python import BaseOptions
    import numpy as np

    _ensure_mp_model()
    h, w = img_rgb.shape[:2]

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        num_poses=1,
    )
    with PoseLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

    kp = np.zeros((N_BODY_KP, 3), dtype=np.float32)
    if not result.pose_landmarks:
        print('[warn] MediaPipe: no person detected, using fallback bbox')
        kp[5]  = [w * 0.4, h * 0.25, 0.5]
        kp[6]  = [w * 0.6, h * 0.25, 0.5]
        kp[11] = [w * 0.4, h * 0.55, 0.5]
        kp[12] = [w * 0.6, h * 0.55, 0.5]
        kp[15] = [w * 0.4, h * 0.90, 0.5]
        kp[16] = [w * 0.6, h * 0.90, 0.5]
        return kp

    lms = result.pose_landmarks[0]
    for coco_i, mp_i in enumerate(_MP_TO_COCO):
        lm = lms[mp_i]
        kp[coco_i] = [lm.x * w, lm.y * h, lm.visibility]
    return kp


def _make_batch(img_rgb, joints2d, img_res=224):
    import numpy as np
    import torch
    from torchvision.transforms import Normalize

    _ensure_pymafx_on_path()
    from core import constants
    from utils.imutils import crop

    h, w = img_rgb.shape[:2]
    valid = joints2d[joints2d[:, 2] > 0.1]
    if len(valid) < 2:
        valid = joints2d

    x1, y1 = valid[:, 0].min(), valid[:, 1].min()
    x2, y2 = valid[:, 0].max(), valid[:, 1].max()
    center = [(x1 + x2) / 2., (y1 + y2) / 2.]
    scale = 1.2 * max(x2 - x1, y2 - y1) / 200.

    res = [img_res, img_res]
    crop_img, _, _ = crop(img_rgb, center, scale, res)
    crop_arr = np.transpose(crop_img.astype('float32'), (2, 0, 1)) / 255.0

    normalize = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img_tensor = normalize(torch.from_numpy(crop_arr).float())
    dummy = torch.zeros(3, img_res, img_res)

    return {
        'img_body': img_tensor.unsqueeze(0),
        'img_lhand': dummy.unsqueeze(0),
        'img_rhand': dummy.unsqueeze(0),
        'img_face': dummy.unsqueeze(0),
        'lhand_theta_inv': torch.zeros(1, 2, 3),
        'rhand_theta_inv': torch.zeros(1, 2, 3),
        'face_theta_inv': torch.zeros(1, 2, 3),
        'vis_lhand': torch.tensor([0.0]),
        'vis_rhand': torch.tensor([0.0]),
        'vis_face': torch.tensor([0.0]),
        'orig_height': torch.tensor([h]),
        'orig_width': torch.tensor([w]),
        'person_id': ['person_0'],
    }


def infer(img_bgr) -> dict:
    """
    Run PyMAF-X on a BGR image (numpy HxWx3).
    Returns dict with:
      betas        (10,)       — SMPL-X shape coefficients
      smplx_verts  (10475, 3)  — posed vertices from PyMAF-X
    """
    import cv2, torch
    import numpy as np

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f'PyMAF-X: image {img_rgb.shape}')

    print('Detecting keypoints...')
    joints2d = detect_body_keypoints(img_rgb)
    vis = (joints2d[:, 2] > 0.3).sum()
    print(f'  Visible: {vis}/{N_BODY_KP}')

    model, device = load_model()
    batch = _make_batch(img_rgb, joints2d)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    orig_cwd = os.getcwd()
    os.chdir(PYMAFX_DIR)
    try:
        with torch.no_grad():
            preds_dict, _ = model(batch)
    finally:
        os.chdir(orig_cwd)

    output = preds_dict['mesh_out'][-1]
    return {
        'betas': output['pred_shape'].squeeze(0).cpu().numpy(),
        'smplx_verts': output['smplx_verts'].squeeze(0).cpu().numpy(),
    }
