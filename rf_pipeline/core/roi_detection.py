from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.ndimage import (
    gaussian_filter, uniform_filter, label, find_objects,
    binary_opening, binary_closing, binary_dilation
)
from scipy.stats import norm


@dataclass
class ROI:
    """Contenedor para una región rectangular de interés (ROI)."""
    y1: int
    x1: int
    y2: int
    x2: int
    score: float
    snr_db: float = 0.0  # opcional (rellenado por etapas previas)


class ROIDetector:
    """
    Step 2 — Low-Cost ROI Detection + CFAR fusion
    Combina umbral adaptativo + 2D CA-CFAR ligero para detectar señales incluso con SNR bajo.
    """

    def __init__(
        self,
        method: str = "adaptive",
        gauss_sigma: float = 0.5,
        min_area: int = 6,
        adaptive_window: int = 11,
        adaptive_k: float = -0.15,
        margin_bins: int = 1,
        min_ar: float = 0.10,
        max_ar: float = 12.0,
        min_texture: float = 0.03,
        # --- CFAR params ---
        use_cfar: bool = True,
        cfar_win: Tuple[int, int] = (13, 25),
        cfar_guard: Tuple[int, int] = (2, 4),
        cfar_pfa: float = 3e-3,
    ):
        self.method = method
        self.gauss_sigma = gauss_sigma
        self.min_area = min_area
        self.adaptive_window = adaptive_window
        self.adaptive_k = adaptive_k
        self.margin_bins = margin_bins
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.min_texture = min_texture

        self.use_cfar = use_cfar
        self.cfar_win = cfar_win
        self.cfar_guard = cfar_guard
        self.cfar_pfa = cfar_pfa

    # -------------------------------------------------------------

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Umbral local adaptativo."""
        w = max(3, int(self.adaptive_window) | 1)
        local_mean = uniform_filter(img, size=w, mode="reflect")
        thr = local_mean + self.adaptive_k * (local_mean - img.min())
        return img > thr

    # -------------------------------------------------------------

    def _cfar_mask(self, img: np.ndarray) -> np.ndarray:
        """
        2D CA-CFAR ligero: calcula umbral local = mu + k*sigma,
        con k derivado de la probabilidad de falsa alarma (Pfa).
        """
        H, W = img.shape
        win_h, win_w = self.cfar_win
        gh, gw = self.cfar_guard
        win_h = (win_h // 2) * 2 + 1
        win_w = (win_w // 2) * 2 + 1

        m = uniform_filter(img, size=(win_h, win_w), mode="reflect")
        m2 = uniform_filter(img**2, size=(win_h, win_w), mode="reflect")
        var = np.maximum(m2 - m**2, 1e-12)

        guard_h, guard_w = (2 * gh + 1), (2 * gw + 1)
        mg = uniform_filter(img, size=(guard_h, guard_w), mode="reflect")
        mg2 = uniform_filter(img**2, size=(guard_h, guard_w), mode="reflect")

        big_area = win_h * win_w
        guard_area = guard_h * guard_w
        ref_area = max(big_area - guard_area, 1)

        mu_ref = (m * big_area - mg * guard_area) / ref_area
        m2_ref = (m2 * big_area - mg2 * guard_area) / ref_area
        var_ref = np.maximum(m2_ref - mu_ref**2, 1e-12)
        sigma_ref = np.sqrt(var_ref)

        k = float(norm.isf(self.cfar_pfa))
        thr = mu_ref + k * sigma_ref

        return img > thr

    # -------------------------------------------------------------

    def detect(self, S: np.ndarray) -> List[ROI]:
        """
        Detección principal de regiones de interés (ROIs) sobre un espectrograma 2D.
        """
        if S.ndim != 2:
            raise ValueError("Expected a 2D spectrogram.")

        img = (S.astype(np.float32) - np.min(S)) / (np.ptp(S) + 1e-12)
        img = gaussian_filter(img, sigma=self.gauss_sigma)

        # Máscara adaptativa
        mask_adp = self._adaptive_threshold(img)

        # CFAR opcional
        if self.use_cfar:
            mask_cfar = self._cfar_mask(img)
            mask = np.logical_or(mask_adp, mask_cfar)
        else:
            mask = mask_adp

        # Limpieza morfológica
        mask = binary_opening(mask)
        mask = binary_closing(mask)
        if self.margin_bins > 0:
            mask = binary_dilation(mask, iterations=self.margin_bins)

        labeled, n = label(mask)
        if n == 0:
            return []

        rois = []
        for slc in find_objects(labeled):
            if not slc:
                continue
            y1, y2 = slc[0].start, slc[0].stop
            x1, x2 = slc[1].start, slc[1].stop
            h, w = (y2 - y1), (x2 - x1)
            area = h * w
            if area < self.min_area:
                continue

            patch = img[y1:y2, x1:x2]
            ar = w / (h + 1e-6)
            texture = float(np.std(patch))

            if not (self.min_ar <= ar <= self.max_ar):
                continue
            if texture < self.min_texture:
                continue

            score = float(np.mean(patch))
            rois.append(ROI(y1, x1, y2, x2, score))

        return sorted(rois, key=lambda r: r.score, reverse=True)


# -------------------------------------------------------------
# Visualización auxiliar
# -------------------------------------------------------------
def draw_rois(ax, S: np.ndarray, rois: List[ROI]) -> None:
    """Dibuja rectángulos y etiquetas sobre el espectrograma."""
    import matplotlib.patches as patches
    for r in rois:
        rect = patches.Rectangle(
            (r.x1, r.y1),
            r.x2 - r.x1, r.y2 - r.y1,
            linewidth=1.2, edgecolor="white", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            r.x1, r.y1 - 1,
            f"{getattr(r, 'snr_db', 0):.1f} dB" if hasattr(r, "snr_db") else f"{r.score:.2f}",
            color="white", fontsize=7, va="top"
        )
