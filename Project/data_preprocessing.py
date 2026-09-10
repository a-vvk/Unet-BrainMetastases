import os
import time
import numpy as np
import cv2

def load_npy_files(directory):
    data = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith('.npy'):
            data[fn] = np.load(os.path.join(directory, fn))
    return data

def normalize_volume(vol):
    vol = vol.astype('float32')
    m = vol.max()
    return vol / m if m > 0 else vol

def extract_slices(vol, axis=2):
    return [np.take(vol, i, axis=axis) for i in range(vol.shape[axis])]

def resize_slices(slices, target_size=(128,128)):
    return [cv2.resize(s, target_size) for s in slices]

def filter_black_pairs(img_slices, msk_slices, blank_thresh=0.0):
    kept_i, kept_m = [], []
    for img, msk in zip(img_slices, msk_slices):
        nonblank = np.count_nonzero(np.any(img > 0, axis=-1))
        total = img.shape[0] * img.shape[1]
        if nonblank / total > blank_thresh:
            kept_i.append(img)
            kept_m.append(msk)
    return kept_i, kept_m

if __name__ == '__main__':
    t0 = time.time()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR   = os.path.dirname(SCRIPT_DIR)
    RAW_ROOT   = os.path.join(BASE_DIR, 'Data', 'Raw')
    PRE_IMG    = os.path.join(BASE_DIR, 'Data', 'Preprocessed', 'Images')
    PRE_MSK    = os.path.join(BASE_DIR, 'Data', 'Preprocessed', 'Masks')
    os.makedirs(PRE_IMG, exist_ok=True)
    os.makedirs(PRE_MSK, exist_ok=True)

    target_size  = (128, 128)
    slice_axis   = 2    
    blank_thresh = 0.0   
    modalities   = ['t1c', 't1n', 't2f', 't2w']

    all_imgs = []
    all_msks = []

    for patient in sorted(os.listdir(RAW_ROOT)):
        patient_dir = os.path.join(RAW_ROOT, patient)
        img_dir = os.path.join(patient_dir, 'Images')
        msk_dir = os.path.join(patient_dir, 'Masks')

        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            print(f"Skipping {patient}: missing subfolder Images/ or Masks/")
            continue

        images = load_npy_files(img_dir)
        masks  = load_npy_files(msk_dir)

        vols = {}
        for mod in modalities:
            key = next((fn for fn in images if fn.endswith(f"-{mod}.npy")), None)
            if key is None:
                print(f"   {patient}: no volume for modality '{mod}'")
                vols = None
                break
            vols[mod] = images[key]
        if vols is None:
            continue

        seg_key = next((fn for fn in masks if fn.endswith("-seg.npy")), None)
        if seg_key is None:
            print(f"   {patient}: no segmentation file ('-seg.npy')")
            continue
        seg_vol = masks[seg_key]

        for mod in modalities:
            vols[mod] = normalize_volume(vols[mod])
        seg_vol = normalize_volume(seg_vol)

        slices_mod = {
            mod: resize_slices(extract_slices(vols[mod], axis=slice_axis), target_size)
            for mod in modalities
        }
        seg_slices = resize_slices(extract_slices(seg_vol, axis=slice_axis), target_size)
        depth = len(seg_slices)

        patient_imgs = []
        patient_msks = []
        for i in range(depth):
            chans = [slices_mod[mod][i] for mod in modalities]
            img4c = np.stack(chans, axis=-1)          
            msk2d = seg_slices[i]                    
            patient_imgs.append(img4c)
            patient_msks.append(msk2d)

        kept_i, kept_m = filter_black_pairs(patient_imgs, patient_msks, blank_thresh)
        all_imgs.extend(kept_i)
        all_msks.extend(kept_m)
        print(f"  {patient}: kept {len(kept_i)}/{depth} non-blank slices")

    X = np.stack(all_imgs, axis=0)                
    Y = np.expand_dims(np.stack(all_msks, axis=0), -1)  
    print(f"\nFinal shapes: X={X.shape}, Y={Y.shape}")

    np.save(os.path.join(PRE_IMG, 'images_preprocessed.npy'), X)
    np.save(os.path.join(PRE_MSK, 'masks_preprocessed.npy'), Y)
    elapsed = time.time() - t0
    print(f"\nSaved to:\n  {PRE_IMG}\n  {PRE_MSK}")
    print(f"Done in {elapsed:.1f}s")