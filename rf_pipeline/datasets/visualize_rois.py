import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

from preprocessing import Preprocessor            # Usa tu pipeline real
from roi_detection import ROIDetector
from visual_stream import VisualStream            # Para normalización visual 1 canal
from feature_engineering import FeatureEngineer

FS = 1_000_000
NPERSEG = 32
NOVERLAP = 16

DETECTOR_KW = dict(
    method="adaptive",
    margin_bins=0,
    gauss_sigma=0.8,
    min_area=20,
    adaptive_window=9,
    adaptive_k=-0.5,
)

TARGET_SIZE = 32
NEG_PER_SIGNAL = 4
MIN_NEG_AREA = 20

def load_iq(fpath):
    iq2 = np.load(fpath)
    return iq2[:,0] + 1j*iq2[:,1]

def find_files(root):
    files = []
    for split in ["train", "val", "test"]:
        files.extend(glob.glob(os.path.join(root, split, "*", "*.npy")))
    return sorted(files)

def non_overlapping_negative(S_mce, rois, max_neg=1):
    C0 = S_mce[0]
    H, W = C0.shape
    results = []
    tries = 0
    while len(results) < max_neg and tries < max_neg * 6:
        tries += 1
        h = random.randint(4, max(4, H//6))
        w = random.randint(4, max(4, W//6))
        y1 = random.randint(0, max(0, H - h))
        x1 = random.randint(0, max(0, W - w))
        y2, x2 = y1 + h, x1 + w

        if any(not (x2 <= r.x1 or r.x2 <= x1 or y2 <= r.y1 or r.y2 <= y1) for r in rois):
            continue

        patch_c0 = C0[y1:y2, x1:x2]
        if patch_c0.size == 0:
            continue

        patch_resized = cv2.resize(patch_c0, (TARGET_SIZE, TARGET_SIZE))
        results.append((patch_resized, (y1,x1,y2,x2)))

    return results

def visualize(root, num_pos=10, num_neg=10):
    pre = Preprocessor(fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, mode="mce")
    det = ROIDetector(**DETECTOR_KW)

    files = find_files(root)
    random.shuffle(files)

    positives = []
    negatives = []

    for f in files:
        iq = load_iq(f)
        S_mce = pre.compute(iq, show=False)
        C0 = S_mce[0]  # espectrograma base log|STFT|

        rois = det.detect(S_mce[1])

        if len(rois) > 0 and len(positives) < num_pos:
            positives.append((C0, rois, f))

        if len(rois) > 0 and len(negatives) < num_neg:
            negs = non_overlapping_negative(S_mce, rois, max_neg=1)
            if negs:
                patch, (y1,x1,y2,x2) = negs[0]
                negatives.append((C0, [(y1,x1,y2,x2)], f))

        if len(positives) >= num_pos and len(negatives) >= num_neg:
            break

    print(f"[OK] Encontrados {len(positives)} POS y {len(negatives)} NEG para mostrar.\n")

    for C0, rois, f in positives:
        plt.figure(figsize=(10,4))
        plt.imshow(C0, aspect="auto", origin="lower", cmap="viridis")
        for r in rois:
            plt.gca().add_patch(plt.Rectangle((r.x1, r.y1), r.x2-r.x1, r.y2-r.y1,
                                              fill=False, edgecolor="lime", linewidth=2))
        plt.title(f"POSITIVO — {os.path.basename(f)}")
        plt.show()

    for C0, rois, f in negatives:
        plt.figure(figsize=(10,4))
        plt.imshow(C0, aspect="auto", origin="lower", cmap="viridis")
        for (y1,x1,y2,x2) in rois:
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                              fill=False, edgecolor="red", linewidth=2))
        plt.title(f"NEGATIVO — {os.path.basename(f)}")
        plt.show()


if __name__ == "__main__":
    visualize(
        root=r"C:\Users\User\Documents\Git\MCE-ROI\rf_dataset",
        num_pos=10,
        num_neg=10
    )
