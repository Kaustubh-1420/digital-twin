"""
smplx-measure Gradio app
Photo + height → personalized SMPL-X avatar (GLB) + body measurements.
Deployed on HF Spaces ZeroGPU.
"""
import os
import sys
import tempfile

# Must be set before any torch/CUDA import on ZeroGPU
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import torch
import cv2
import smplx
import trimesh
import gradio as gr

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'backend'))

from pymafx_backend import infer as pymafx_infer, load_model as pymafx_load
from scale import scale_to_height
from measurements import extract_measurements

SMPLX_MODEL_PATH = os.path.join(ROOT, 'models')

# ── ZeroGPU ───────────────────────────────────────────────────────────────────
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False
    # local shim so decorator syntax works without error
    class _spaces:
        @staticmethod
        def GPU(fn=None, duration=120):
            return fn if fn else (lambda f: f)
    spaces = _spaces()


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_smplx():
    return smplx.create(
        model_path=SMPLX_MODEL_PATH,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        num_betas=10,
        batch_size=1,
    )


def _mesh_to_glb(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Export mesh to a temp GLB file. Returns the file path."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    tmp.close()
    mesh.export(tmp.name, file_type='glb')
    return tmp.name


def _fmt_measurements(m: dict) -> str:
    labels = {
        'chest_cm':      'Chest',
        'waist_cm':      'Waist',
        'hip_cm':        'Hip',
        'shoulder_cm':   'Shoulder width',
        'inseam_cm':     'Inseam',
        'arm_length_cm': 'Arm length',
    }
    lines = []
    for key, label in labels.items():
        val = m.get(key)
        if val is not None:
            lines.append(f'{label:<18} {val:.1f} cm')
    return '\n'.join(lines)


# ── core pipeline ─────────────────────────────────────────────────────────────

@spaces.GPU(duration=120)
def run_pipeline(image_path: str, height_cm: float):
    """
    image_path: path written by Gradio Image component (filepath mode)
    height_cm:  user height in centimetres
    Returns: (glb_path, measurements_text, status_text)
    """
    if image_path is None:
        return None, '', 'Please upload a photo.'

    height_cm = float(height_cm)
    if not (100 <= height_cm <= 250):
        return None, '', 'Height must be between 100 and 250 cm.'

    # 1. load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, '', 'Could not read image — try a JPG or PNG.'

    # 2. PyMAF-X inference → native SMPL-X betas
    result = pymafx_infer(img_bgr)
    betas = torch.from_numpy(result['betas']).unsqueeze(0).float()

    # 3. T-pose with estimated betas
    smplx_model = _build_smplx()
    out = smplx_model(
        betas=betas,
        body_pose=torch.zeros(1, 63),
        global_orient=torch.zeros(1, 3),
        return_verts=True,
    )
    vertices = out.vertices[0].detach().numpy()
    joints   = out.joints[0].detach().numpy()
    faces    = smplx_model.faces

    # 4. scale to user height
    vertices, joints = scale_to_height(vertices, height_cm, joints)

    # 5. measurements
    measurements = extract_measurements(vertices, faces, joints)
    meas_text = _fmt_measurements(measurements)

    # 6. export GLB
    glb_path = _mesh_to_glb(vertices, faces)

    return glb_path, meas_text, '✓ Done'


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#meas  { font-family: monospace; font-size: 14px; white-space: pre; }
"""

with gr.Blocks(css=CSS, title='smplx-measure') as demo:
    gr.Markdown('# smplx-measure\n**Photo → personalized SMPL-X body avatar + measurements**',
                elem_id='title')

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label='Your photo (single person, full body preferred)',
                type='filepath',
            )
            height_slider = gr.Slider(
                minimum=100, maximum=250, value=170, step=1,
                label='Your height (cm)',
            )
            run_btn = gr.Button('Estimate body →', variant='primary')
            status_box = gr.Textbox(label='Status', interactive=False, lines=1)

        with gr.Column(scale=2):
            model3d = gr.Model3D(
                label='Your avatar (T-pose)',
                clear_color=[0.15, 0.15, 0.15, 1.0],
            )
            meas_box = gr.Textbox(
                label='Measurements',
                interactive=False,
                lines=8,
                elem_id='meas',
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[img_input, height_slider],
        outputs=[model3d, meas_box, status_box],
    )

    gr.Markdown(
        '**Note:** measurements are estimates from a single photo. '
        'Accuracy improves with a full-body, front-facing photo in fitted clothing. '
        'First run may take ~30 s while the GPU warms up.'
    )


if __name__ == '__main__':
    # pre-load model so first request is faster
    print('Pre-loading PyMAF-X...')
    pymafx_load()
    demo.launch(share=False)
