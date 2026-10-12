"""
digital-twin Gradio app
Photo + height → personalized SMPL-X avatar (GLB) + body measurements.
Deployed on HF Spaces ZeroGPU.
"""
import os
import sys
import subprocess
import time

# ── run setup.sh if assets are missing (HF Spaces may not auto-run it) ──────
ROOT = os.path.dirname(os.path.abspath(__file__))
NLF_PATH = '/tmp/nlf/nlf_l_multi_0.3.2.torchscript'

def _run_setup():
    if not (os.path.exists('/tmp/smplx-models/smplx/SMPLX_NEUTRAL.npz') and os.path.exists(NLF_PATH)):
        setup = os.path.join(ROOT, 'setup.sh')
        print('Running setup.sh...')
        subprocess.run(['bash', setup], check=True)
    else:
        print('Assets present, skipping setup.')

_run_setup()

import numpy as np
import torch
import torchvision  # noqa: F401  (registers ops the NLF TorchScript model needs)
import cv2
import smplx
import gradio as gr

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(ROOT, 'backend'))

from scale import scale_to_height
from measurements import extract_measurements
from export_glb import export_skinned_glb, compute_expression_morphs

SMPLX_MODEL_PATH = (
    '/tmp/smplx-models'
    if os.path.isdir('/tmp/smplx-models/smplx')
    else os.path.join(ROOT, 'models')
)

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


# NLF's TorchScript otherwise profiles and re-optimises its graph on the first
# calls in each process (~45 s on the first image; measured on a T4). Same
# outputs; first image ~4 s. Set again inside the GPU call in case ZeroGPU's
# worker process doesn't inherit it.
torch._C._set_graph_executor_optimize(False)

# NLF is loaded per call inside the GPU function. A startup load with
# map_location='cuda' bypasses ZeroGPU's CUDA emulation, fails with "no
# CUDA-capable device", and that failed init then breaks every GPU worker.


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_smplx():
    return smplx.create(
        model_path=SMPLX_MODEL_PATH,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=100,
        batch_size=1,
    )


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

@spaces.GPU(duration=20)
def _estimate_betas(img_rgb: np.ndarray):
    """NLF → SMPL-X betas (1, 10) on CPU, or None if no person is found.
    Only this runs on the GPU; everything after it is CPU work."""
    t0 = time.perf_counter()
    torch._C._set_graph_executor_optimize(False)
    # map_location='cuda': ZeroGPU can't intercept a load-time .to('cuda'),
    # and .to() would miss tensors the model keeps in plain dict attributes.
    nlf = torch.jit.load(NLF_PATH, map_location='cuda').eval()
    t_load = time.perf_counter()
    img = torch.from_numpy(img_rgb).permute(2, 0, 1).cuda()
    with torch.inference_mode():
        pred = nlf.detect_smpl_batched(img[None], model_name='smplx')
    people = pred['betas'][0]
    betas = None
    if people.shape[0] > 0:
        # Several people: take the largest by 2D joint extent
        j2d = pred['joints2d'][0]
        best = int((j2d.amax(1) - j2d.amin(1)).prod(-1).argmax())
        betas = people[best:best + 1].float().cpu()
    t_end = time.perf_counter()
    print(f'[timing] gpu: load {t_load - t0:.1f}s, infer {t_end - t_load:.1f}s')
    return betas


def run_pipeline(image_path: str, height_cm: float):
    """
    image_path: path written by Gradio Image component (filepath mode)
    height_cm:  user height in centimetres
    Returns: (glb_path, measurements_text, status_text)
    """
    t0 = time.perf_counter()
    if image_path is None:
        return None, '', 'Please upload a photo.'

    height_cm = float(height_cm)
    if not (100 <= height_cm <= 250):
        return None, '', 'Height must be between 100 and 250 cm.'

    # 1. load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, '', 'Could not read image — try a JPG or PNG.'

    # 2. NLF → SMPL-X betas (the only GPU step)
    t_gpu = time.perf_counter()
    betas = _estimate_betas(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t_cpu = time.perf_counter()
    if betas is None:
        return None, '', 'No person found in the photo. Use a full-body photo of one person.'

    # 3. T-pose with estimated betas
    smplx_model = _build_smplx()
    out = smplx_model(
        betas=betas,
        body_pose=torch.zeros(1, 63),
        global_orient=torch.zeros(1, 3),
        return_verts=True,
    )
    raw_verts = out.vertices[0].detach().numpy()
    joints    = out.joints[0].detach().numpy()
    faces     = smplx_model.faces

    # 4. scale to user height
    mesh_h = float(raw_verts[:, 1].max() - raw_verts[:, 1].min())
    scale_factor = (height_cm / 100.0) / mesh_h
    vertices = (raw_verts * scale_factor).astype(raw_verts.dtype)
    joints   = (joints    * scale_factor).astype(joints.dtype)

    # 5. measurements
    measurements = extract_measurements(vertices, faces, joints)
    meas_text = _fmt_measurements(measurements)

    # 6. expression morph targets — user-specific betas, scaled to match mesh
    morph_deltas = compute_expression_morphs(smplx_model, betas, scale_factor)

    # 7. export skinned GLB with VRM bone names and face morph targets
    lbs_weights = smplx_model.lbs_weights.detach().numpy()
    glb_path = export_skinned_glb(vertices, faces, joints, lbs_weights, morph_deltas)

    t_end = time.perf_counter()
    print(f'[timing] run_pipeline: gpu call {t_cpu - t_gpu:.1f}s, cpu {t_end - t_cpu:.1f}s, '
          f'total {t_end - t0:.1f}s')
    return glb_path, meas_text, '✓ Done'


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; }
#meas  { font-family: monospace; font-size: 14px; white-space: pre; }
"""

with gr.Blocks(title='digital-twin') as demo:
    gr.Markdown('# digital-twin\n**Photo → personalized SMPL-X body avatar + measurements**',
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
            model3d = gr.File(
                label='Avatar GLB (download or use via frontend)',
                file_types=['.glb'],
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
    print(f'Python {sys.version.split()[0]}, torch {torch.__version__}, '
          f'CUDA build {torch.version.cuda}, cuda available {torch.cuda.is_available()}, '
          f'ZeroGPU {HAS_ZEROGPU}')
    demo.launch(css=CSS, share=False)
