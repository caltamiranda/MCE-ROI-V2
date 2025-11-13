# rf_pipeline/feature_engineering.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

from roi_detection import ROI


@dataclass
class ROIFeatures:
    """Simple container: one ROI + its 32-D feature vector."""
    roi: ROI
    features: np.ndarray  # shape (32,)


class FeatureEngineer:
    """
    Step 3A — Engineered Feature Stream (per ROI)

    - Extracts a raw time-domain slice mapped from each ROI's time-bin span in S (STFT).
    - Works on |I + jQ| magnitude only (real-valued).
    - Wavelet-based compression (db4) + robust statistics.
    - Returns exactly 32-D feature vectors (fixed size).
    """

    def __init__(self, fs: float = 1e6, nperseg: int = 32, noverlap: int = 16):
        """
        Args:
            fs       : Sampling rate used during STFT.
            nperseg  : STFT window length (must match Preprocessor).
            noverlap : STFT overlap (same as Preprocessor).
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.hop = self.nperseg - self.noverlap  # hop size = time-bin stride in STFT

    # -------------------------------------------------------------

    def _roi_time_samples(self, roi: ROI, S_shape: tuple, sig_len: int) -> Optional[slice]:
        """
        Map ROI time-bin indices (x1..x2) to a 1D sample slice of the raw I/Q signal.

        Approximation:
            STFT advances by 'hop' samples per time column.
            We cover start at x1 * hop, and end at (x2 - 1) * hop + nperseg.
        """
        _, W = S_shape  # freq_bins, time_bins
        x1, x2 = roi.x1, roi.x2

        # Clamp into valid index range
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))

        if x2 <= x1:
            return None

        start = int(x1 * self.hop)
        end = int((x2 - 1) * self.hop + self.nperseg)
        start = max(0, start)
        end = min(sig_len, end)

        # Reject overly short segments (unstable stats)
        if end - start < max(8, self.nperseg // 2):
            return None

        return slice(start, end)

    # -------------------------------------------------------------
    # Low-level feature helpers
    # -------------------------------------------------------------

    @staticmethod
    def _safe_entropy(x: np.ndarray, bins: int = 64) -> float:
        """Shannon entropy with histogram-based probability — returns 0 if flat."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0 or np.allclose(x, 0):
            return 0.0
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = hist / (np.sum(hist) + 1e-12)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _zcr(x: np.ndarray) -> float:
        """Zero-crossing rate (binary crossing of sign)."""
        return float(np.mean(np.abs(np.diff(np.signbit(x)).astype(np.float32))))

    @staticmethod
    def _tk_energy(x: np.ndarray) -> float:
        """Mean Teager–Kaiser energy operator."""
        if x.size < 3:
            return 0.0
        psi = x[1:-1] ** 2 - x[0:-2] * x[2:]
        return float(np.mean(np.abs(psi)))

    @staticmethod
    def _crest_factor(x: np.ndarray) -> float:
        """Peak-to-RMS ratio."""
        rms = np.sqrt(np.mean(x**2) + 1e-12)
        peak = np.max(np.abs(x)) + 1e-12
        return float(peak / rms)

    # -------------------------------------------------------------
    # Main blocks
    # -------------------------------------------------------------

    def _wavelet_stats(self, xmag: np.ndarray, snr_db: Optional[float] = None) -> Dict[str, Any]:
        """
        Wavelet decomposition adaptativa según SNR.

        - Usa 'sym5' en bajo SNR (<3 dB), 'db2' en medio SNR (3–10 dB) y 'db4' en alto SNR (>10 dB).
        - Extrae energías y estadísticas detalladas de coeficientes.
        """
        # Selección adaptativa de wavelet
        if snr_db is None:
            wavelet_name = 'db4'
        elif snr_db < 3:
            wavelet_name = 'sym5'
        elif snr_db < 10:
            wavelet_name = 'db2'
        else:
            wavelet_name = 'db4'

        w = pywt.Wavelet(wavelet_name)
        max_level = min(5, pywt.dwt_max_level(len(xmag), w.dec_len))
        coeffs = pywt.wavedec(xmag, w, level=max_level)

        # Energía por subbanda
        energies = [np.sum(c**2) for c in coeffs]
        total_e = np.sum(energies) + 1e-12

        det_energies = [np.sum(c**2) for c in coeffs[1:]]
        approx_energy = np.sum(coeffs[0] ** 2)
        vec = np.array(det_energies + [approx_energy], dtype=np.float32) / total_e
        vec = np.pad(vec, (0, max(0, 8 - vec.size)))[:8]

        # Estadísticas de detalle
        detail_stats = []
        for i in range(1, min(5, len(coeffs))):
            cd = np.abs(coeffs[-i])
            detail_stats.extend([float(np.mean(cd)), float(np.std(cd) + 1e-12)])
            if len(detail_stats) >= 8:
                break
        if len(detail_stats) < 8:
            detail_stats.extend([0.0] * (8 - len(detail_stats)))

        return {
            "energy8": vec,
            "detail_stats8": np.array(detail_stats, dtype=np.float32),
            "wavelet_used": wavelet_name,
        }

    def _stats_block(self, xmag: np.ndarray) -> np.ndarray:
        """
        Core statistical descriptors (16 dims).
        """
        x = xmag.astype(np.float32)
        if x.size == 0:
            return np.zeros(16, dtype=np.float32)

        q1, med, q3 = np.percentile(x, [25, 50, 75])
        iqr = q3 - q1
        rms = np.sqrt(np.mean(x**2) + 1e-12)
        var = np.var(x)
        std = np.sqrt(var + 1e-12)

        feats = [
            rms,
            var,
            float(skew(x, bias=False)) if x.size > 3 else 0.0,
            float(kurtosis(x, fisher=False, bias=False)) if x.size > 3 else 3.0,
            self._safe_entropy(x),
            float(q1),
            float(med),
            float(q3),
            iqr,
            std,
            float(np.mean(np.abs(x - np.mean(x)))),  # MAD
            self._zcr(x),
            self._tk_energy(x),
            self._crest_factor(x),
            float(np.min(x)),
            float(np.max(x)),
        ]
        return np.array(feats, dtype=np.float32)

    # -------------------------------------------------------------

    def features_for_rois(self, iq_signal: np.ndarray, S: np.ndarray, rois: List[ROI]) -> List[ROIFeatures]:
        """
        Generate one 32-D feature vector per valid ROI.

        Output shape per ROI:
            [16 stats] + [8 wavelet energy] + [8 wavelet detail stats] = 32
        """
        results: List[ROIFeatures] = []
        sig_len = len(iq_signal)

        for roi in rois:
            sl = self._roi_time_samples(roi, S.shape, sig_len)
            if sl is None:
                continue

            # Extract and convert to magnitude
            seg = iq_signal[sl]
            xmag = np.abs(seg).astype(np.float32)

            stats16 = self._stats_block(xmag)
            wav = self._wavelet_stats(xmag)
            wave16 = np.concatenate([wav["energy8"], wav["detail_stats8"]], axis=0)

            feat32 = np.concatenate([stats16, wave16], axis=0).astype(np.float32)
            feat32 = np.nan_to_num(feat32, nan=0.0, posinf=0.0, neginf=0.0)

            results.append(ROIFeatures(roi=roi, features=feat32))

        return results
