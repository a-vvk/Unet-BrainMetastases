import os, numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

def list_patients(raw_root):
    return sorted([d for d in os.listdir(raw_root) 
                   if os.path.isdir(os.path.join(raw_root, d))])

def split_patients(patients, val_frac=0.2, seed=42):
    train, val = train_test_split(patients, test_size=val_frac, random_state=seed)
    return train, val

class PatientSliceSequence(Sequence):
    def __init__(self,
                 raw_root,                 
                 patient_ids,               
                 batch_size=8,
                 target_size=(128,128),
                 slice_axis=2,
                 blank_thresh=0.15,         
                 augment_fn=None,
                 shuffle=True):
        self.raw_root      = raw_root
        self.patients      = patient_ids
        self.batch_size    = batch_size
        self.target_size   = target_size
        self.axis          = slice_axis
        self.blank_thresh  = blank_thresh
        self.augment_fn    = augment_fn
        self.shuffle       = shuffle

        self.volumes = {}
        for pid in self.patients:
            img = np.load(os.path.join(raw_root, pid, "image.npy"))
            msk = np.load(os.path.join(raw_root, pid, "mask.npy"))
            self.volumes[pid] = {
                "img": img.astype("float32") / np.max(img),
                "msk": (msk>0).astype("float32")
            }

        self.indexes = []
        for pid in self.patients:
            D = self.volumes[pid]["img"].shape[self.axis]
            for s in range(D):
                self.indexes.append((pid, s))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        Xs, Ys = [], []
        for pid, sl in batch:
            img_vol = self.volumes[pid]["img"]
            msk_vol = self.volumes[pid]["msk"]
            slc = np.take(img_vol, sl, axis=self.axis)
            msk = np.take(msk_vol, sl, axis=self.axis)

            if np.count_nonzero(slc)/slc.size < self.blank_thresh:
                continue

            slc = tf.image.resize(slc[...,None], self.target_size).numpy()
            msk = tf.image.resize(msk[...,None], self.target_size, method="nearest").numpy()

            if self.augment_fn:
                slc, msk = self.augment_fn(slc, msk)

            Xs.append(slc)
            Ys.append(msk)

        X = np.stack(Xs,0)
        Y = np.stack(Ys,0)
        return X, Y