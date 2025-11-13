# rf_pipeline/visual_stream.py

import numpy as np
import cv2
import matplotlib.pyplot as plt


class VisualStream:
    """
    Step 3B — Visual patches (per ROI)
    - Soporta S 2D (H×W) o 3D (C×H×W). Devuelve [C, target, target] o [1, target, target].
    """

    def __init__(self, target_size: int = 32):
        self.target_size = target_size

    def _norm_patch(self, x: np.ndarray) -> np.ndarray:
        x = np.log1p(x.astype(np.float32))
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    def extract_patches(self, S: np.ndarray, rois, preview: bool = False):
        patches = []
        if S.ndim == 2:
            # 1 canal → reusa pipeline existente
            for i, r in enumerate(rois):
                patch = S[r.y1:r.y2, r.x1:r.x2]
                if patch.size == 0: patches.append(None); continue
                pr = cv2.resize(patch, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                pf = self._norm_patch(pr)[None, ...]  # [1,H,W]
                patches.append(pf)
                if preview:
                    plt.figure(figsize=(2,2)); plt.imshow(pf[0], cmap='viridis', origin='lower')
                    plt.title(f"ROI {i} — {self.target_size}×{self.target_size}"); plt.axis('off'); plt.show()
            return patches

        if S.ndim == 3:
            C, H, W = S.shape
            for i, r in enumerate(rois):
                chans = []
                for c in range(C):
                    patch = S[c, r.y1:r.y2, r.x1:r.x2]
                    if patch.size == 0: chans=[]; break
                    pr = cv2.resize(patch, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                    chans.append(self._norm_patch(pr))
                if not chans:
                    patches.append(None); continue
                pf = np.stack(chans, axis=0).astype(np.float32)  # [C,H,W]
                patches.append(pf)
                if preview:
                    fig, axs = plt.subplots(1, min(3, pf.shape[0]), figsize=(6,2))
                    for k in range(min(3, pf.shape[0])): axs[k].imshow(pf[k], cmap='viridis', origin='lower'); axs[k].axis('off')
                    plt.suptitle(f"ROI {i} — {self.target_size}×{self.target_size}×{pf.shape[0]}"); plt.tight_layout(); plt.show()
            return patches

        raise ValueError("S must be 2D or 3D (C×H×W)")
