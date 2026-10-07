"""
Task 10: validate PyMAF-X pipeline on 10 diverse photos.
Run from project root: python backend/validate_10.py
Downloads test images if not present.
"""
import sys, os, ssl, urllib.request, json, time
import certifi
import cv2
import numpy as np
import torch

sys.path.insert(0, 'backend')

IMAGES_DIR = '/tmp/smplx_test_images'
os.makedirs(IMAGES_DIR, exist_ok=True)

# Public-domain test images (Wikimedia Commons / Pexels — single-person, upright)
REMOTE_IMAGES = [
    # (filename, url)
    ('man_standing_01.jpg',
     'https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/220px-Gatto_europeo4.jpg'),  # fallback placeholder
]

# We'll use locally available images + download a few more via certifi-patched urllib
PUBLIC_IMAGES = [
    ('person_yoga.jpg',
     'https://images.pexels.com/photos/317157/pexels-photo-317157.jpeg?w=640'),
    ('person_standing_m.jpg',
     'https://images.pexels.com/photos/1464625/pexels-photo-1464625.jpeg?w=640'),
    ('person_standing_f.jpg',
     'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?w=640'),
    ('person_sport.jpg',
     'https://images.pexels.com/photos/416778/pexels-photo-416778.jpeg?w=640'),
    ('person_walking.jpg',
     'https://images.pexels.com/photos/1187578/pexels-photo-1187578.jpeg?w=640'),
    ('person_casual.jpg',
     'https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg?w=640'),
    ('person_outdoor.jpg',
     'https://images.pexels.com/photos/1300402/pexels-photo-1300402.jpeg?w=640'),
]

LOCAL_IMAGES = [
    ('/tmp/4D-Humans/example_data/images/pexels-anete-lusina-4793258.jpg', 'hmr2_example'),
    ('/Users/work/personal-projects/smplx-measure/PyMAF-X/examples/coco_images/COCO_val2014_000000477655.jpg', 'coco_01'),
    ('/Users/work/personal-projects/smplx-measure/PyMAF-X/examples/coco_images/COCO_val2014_000000004700.jpg', 'coco_02'),
]

PLAUSIBLE_RANGES = {
    'chest_cm':       (70, 140),
    'waist_cm':       (55, 130),
    'hip_cm':         (70, 145),
    'shoulder_cm':    (25, 55),
    'inseam_cm':      (60, 110),
    'arm_length_cm':  (40, 80),
}


def download_image(url, path):
    ctx = ssl.create_default_context(cafile=certifi.where())
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r, open(path, 'wb') as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f'  [download failed] {e}')
        return False


def prepare_images():
    images = []

    # local first
    for path, name in LOCAL_IMAGES:
        if os.path.exists(path):
            images.append((path, name))

    # download remote
    for fname, url in PUBLIC_IMAGES:
        dest = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(dest):
            print(f'Downloading {fname}...')
            ok = download_image(url, dest)
            if not ok:
                continue
        img = cv2.imread(dest)
        if img is not None and img.shape[0] > 50:
            images.append((dest, fname.replace('.jpg', '')))

    return images[:10]


def check_plausibility(measurements):
    issues = []
    for key, (lo, hi) in PLAUSIBLE_RANGES.items():
        val = measurements.get(key)
        if val is None:
            issues.append(f'{key}: MISSING')
        elif not (lo <= val <= hi):
            issues.append(f'{key}: {val:.1f} out of [{lo}, {hi}]')
    return issues


def run_pipeline(img_bgr, label, height_cm=175.0):
    from pymafx_backend import infer as pymafx_infer
    from scale import scale_to_height
    from measurements import extract_measurements
    import smplx

    t0 = time.time()
    result = pymafx_infer(img_bgr)
    betas = torch.from_numpy(result['betas']).unsqueeze(0).float()

    smplx_model = smplx.create(
        model_path='models', model_type='smplx',
        gender='neutral', use_pca=False, num_betas=10, batch_size=1,
    )
    out = smplx_model(
        betas=betas,
        body_pose=torch.zeros(1, 63),
        global_orient=torch.zeros(1, 3),
        return_verts=True,
    )
    vertices = out.vertices[0].detach().numpy()
    joints   = out.joints[0].detach().numpy()
    faces    = smplx_model.faces

    vertices, joints = scale_to_height(vertices, height_cm, joints)
    measurements = extract_measurements(vertices, faces, joints)
    elapsed = time.time() - t0

    return measurements, elapsed


def main():
    print('=== Task 10: Validate PyMAF-X on 10 photos ===\n')
    images = prepare_images()
    print(f'Found {len(images)} images\n')

    if len(images) == 0:
        print('ERROR: no images found')
        sys.exit(1)

    results = []
    for i, (path, label) in enumerate(images):
        print(f'[{i+1}/{len(images)}] {label}')
        img = cv2.imread(path)
        if img is None:
            print(f'  SKIP: cannot read {path}')
            results.append({'label': label, 'status': 'SKIP', 'measurements': {}, 'issues': []})
            continue

        try:
            measurements, elapsed = run_pipeline(img, label)
            issues = check_plausibility(measurements)
            status = 'FAIL' if issues else 'PASS'
            print(f'  Status: {status}  ({elapsed:.1f}s)')
            for k, v in measurements.items():
                flag = ' !' if any(k in iss for iss in issues) else ''
                print(f'    {k}: {v:.1f} cm{flag}')
            if issues:
                for iss in issues:
                    print(f'    [ISSUE] {iss}')
            results.append({'label': label, 'status': status,
                            'measurements': measurements, 'issues': issues, 'time': elapsed})
        except Exception as e:
            import traceback
            print(f'  CRASH: {e}')
            traceback.print_exc()
            results.append({'label': label, 'status': 'CRASH', 'measurements': {}, 'issues': [str(e)]})
        print()

    # summary
    passed  = sum(1 for r in results if r['status'] == 'PASS')
    failed  = sum(1 for r in results if r['status'] == 'FAIL')
    crashed = sum(1 for r in results if r['status'] == 'CRASH')
    skipped = sum(1 for r in results if r['status'] == 'SKIP')

    print('=== SUMMARY ===')
    print(f'PASS: {passed}  FAIL: {failed}  CRASH: {crashed}  SKIP: {skipped}  / {len(results)} total')

    # print per-measurement stats across passing runs
    passing = [r for r in results if r['status'] in ('PASS', 'FAIL') and r['measurements']]
    if passing:
        print('\nMeasurement stats across all runs:')
        for key in PLAUSIBLE_RANGES:
            vals = [r['measurements'][key] for r in passing if key in r['measurements']]
            if vals:
                print(f'  {key}: min={min(vals):.1f}  max={max(vals):.1f}  mean={np.mean(vals):.1f}')

    # save results
    out_path = '/tmp/validate_10_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nFull results → {out_path}')


if __name__ == '__main__':
    main()
