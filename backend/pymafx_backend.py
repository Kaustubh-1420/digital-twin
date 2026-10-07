"""
PyMAF-X backend — native SMPL-X betas from a single photo.
Must be run with CWD = project root (digital-twin/).

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
    import types, numpy as np

    # chumpy uses `imp` (removed 3.12+) and can't be installed on Python 3.13.
    # pkl files contain pickled chumpy.Ch objects (numpy array subclass) —
    # need a real ndarray subclass so pickle __reduce__ round-trips correctly.
    # Install a catch-all import hook for any chumpy.* submodule so pickle
    # never hits ModuleNotFoundError regardless of which submodule is referenced.
    import importlib.abc, importlib.machinery
    class _ChumPyLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            return mod
        def exec_module(self, mod):
            pass  # populated after _Ch is defined below

    class _ChumPyFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == 'chumpy' or name.startswith('chumpy.'):
                if name not in sys.modules:
                    return importlib.machinery.ModuleSpec(name, _ChumPyLoader())
            return None

    if not any(isinstance(f, _ChumPyFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ChumPyFinder())

    if 'chumpy' not in sys.modules:
        class _Ch:
            """Minimal chumpy.Ch shim — handles SMPL .pkl pickle deserialization.

            Real chumpy.Ch is a plain Python class (not ndarray). Pickle calls
            __new__(cls) with no args, then __setstate__ with chumpy's dict state
            (key 'x' holds the raw numpy array). We implement __array__ so numpy
            treats instances transparently as arrays.
            """
            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)

            def __init__(self, x=None, *args, **kwargs):
                if x is None:
                    self._r = np.empty(0)
                else:
                    try:
                        self._r = np.asarray(x)
                    except Exception:
                        self._r = np.empty(0)

            def __setstate__(self, state):
                # chumpy stores {'x': array_value, ...} or full __dict__
                try:
                    if isinstance(state, dict):
                        x = state.get('x', state.get('r', state.get('_r', None)))
                        self._r = np.asarray(x) if x is not None else np.empty(0)
                    else:
                        self._r = np.empty(0)
                except Exception:
                    self._r = np.empty(0)

            def __array__(self, dtype=None, copy=None):
                return self._r if dtype is None else self._r.astype(dtype)

            # Special methods not caught by __getattr__ — delegate explicitly
            def __getitem__(self, key):   return self._r[key]
            def __setitem__(self, key, v): self._r[key] = v
            def __len__(self):            return len(self._r)
            def __iter__(self):           return iter(self._r)
            def __float__(self):          return float(self._r)
            def __int__(self):            return int(self._r)

            @property
            def r(self):    return self._r
            @property
            def shape(self): return self._r.shape
            @property
            def T(self):     return self._r.T
            @property
            def ndim(self):  return self._r.ndim
            @property
            def dtype(self): return self._r.dtype
            @property
            def size(self):  return self._r.size

            def __getattr__(self, name):
                # Catch-all: delegate attribute lookups to the underlying array.
                # Guard against infinite recursion before _r is set.
                if name == '_r':
                    raise AttributeError(name)
                return getattr(self._r, name)

        # chumpy.reordering.Select: evaluates a.ravel()[idxs].reshape(preferred_shape).
        # Confirmed format from inspecting MANO_RIGHT.pkl with real chumpy:
        #   state = {'a': _Ch(x=ndarray(778,3,20)), 'idxs': ndarray(23340,), 'preferred_shape': (778,3,10), ...}
        class _Select(_Ch):
            def __setstate__(self, state):
                try:
                    a              = state.get('a')
                    idxs           = state.get('idxs')
                    preferred_shape = state.get('preferred_shape')
                    src = np.array(a)           # calls a.__array__() → a._r
                    result = src.ravel()[idxs]  # flat indexing
                    if preferred_shape is not None:
                        result = result.reshape(preferred_shape)
                    self._r = result
                except Exception:
                    self._r = np.empty(0)

        def _make_chumpy_mod(name):
            m = types.ModuleType(name)
            m.Ch = _Ch
            m.array = np.array
            m.__path__ = []
            sys.modules[name] = m
            return m

        _chumpy = _make_chumpy_mod('chumpy')
        # Pre-register every known chumpy submodule; the import hook above
        # catches any others that appear in pickle streams at runtime.
        for _sub in ['chumpy.ch', 'chumpy.reordering', 'chumpy.utils',
                     'chumpy.optimization', 'chumpy.linalg', 'chumpy.logic',
                     'chumpy.indexed_inputs', 'chumpy.check_derivatives']:
            _sm = _make_chumpy_mod(_sub)
            setattr(_chumpy, _sub.split('.')[1], _sm)

        # Attach Select (and common aliases) to reordering
        _chumpy.reordering.Select = _Select
        _chumpy.reordering.Reorder = _Select  # same pattern, different name
        _chumpy.ch.Select = _Select

    # pyrender/neural_renderer/openpifpaf not needed for inference
    # remainder were removed from stdlib in Python 3.12/3.13
    _mocks = [
        'pyrender', 'neural_renderer', 'openpifpaf',
        'cgi', 'cgitb', 'imp', 'aifc', 'audioop', 'chunk', 'crypt',
        'imghdr', 'mailcap', 'msilib', 'nis', 'nntplib', 'ossaudiodev',
        'pipes', 'sndhdr', 'spwd', 'sunau', 'telnetlib', 'uu', 'xdrlib',
    ]
    for mod in _mocks:
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
