import os
import time
import nibabel as nib
import numpy as np

def find_volumes(src_dir):
    for fn in sorted(os.listdir(src_dir)):
        fn_low = fn.lower()
        if fn_low.endswith('.nii.gz'):
            base = fn[:-7]      
        elif fn_low.endswith('.nii'):
            base = fn[:-4]      
        else:
            continue

        is_mask = base.endswith('-seg')
        yield fn, os.path.join(src_dir, fn), base, is_mask

if __name__ == '__main__':
    t0 = time.time()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT    = os.path.dirname(SCRIPT_DIR)
    ORIG_ROOT  = os.path.join(PROJECT, 'Data', 'Original')
    RAW_ROOT   = os.path.join(PROJECT, 'Data', 'Raw')
    os.makedirs(RAW_ROOT, exist_ok=True)

    nii_count = 0
    for idx, orig_patient in enumerate(sorted(os.listdir(ORIG_ROOT)), start=1):
        patient_src = os.path.join(ORIG_ROOT, orig_patient)
        if not os.path.isdir(patient_src):
            continue

        new_patient_name = f"Patient {idx}"
        img_out = os.path.join(RAW_ROOT, new_patient_name, 'Images')
        msk_out = os.path.join(RAW_ROOT, new_patient_name, 'Masks')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(msk_out, exist_ok=True)

        print(f"\nConverting {orig_patient} → {new_patient_name}")

        for fn, fullpath, base, is_mask in find_volumes(patient_src):
            nii_count += 1
            vol = nib.load(fullpath).get_fdata()

            out_fn = base + '.npy'
            out_path = (msk_out if is_mask else img_out) + os.sep + out_fn

            np.save(out_path, vol.astype(np.float32))
            kind = 'MASK' if is_mask else 'IMG '
            print(f"  [{kind}] {fn} → {out_path}")

    print(f"\nDone! Converted {nii_count} volumes in {time.time() - t0:.1f}s")