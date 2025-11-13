# rf_pipeline/preprocessing.py

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class Preprocessor:
    """
    Step 1 — Time-domain I/Q → Time–Frequency representation.

    mode="stft" → 1 canal (actual)
    mode="mce"  → 3 canales (Li et al.): [log|STFT|, gradient, local energy]
    """

    def __init__(self, fs: float = 1e6, nperseg: int = 32, noverlap: int = 16, mode: str = "stft"):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.mode = mode.lower()

    def _stft_logmag(self, iq_signal: np.ndarray) -> np.ndarray:
        f, t, Zxx = stft(iq_signal, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        S = np.abs(Zxx).astype(np.float32)
        S = np.log1p(S)  # canal base
        return S

    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return (x - x.min()) / (x.ptp() + 1e-12)

    def _mce_3ch(self, S: np.ndarray) -> np.ndarray:
        """
        Construye MCE: C0=log|S|, C1=gradiente, C2=energía local.
        Devuelve tensor [3, H, W].
        """
        # C0: base normalizada
        C0 = self._minmax01(S)

        # C1: gradiente (realce de bordes espectrales)
        #   |∂/∂t| + |∂/∂f|
        dt = np.abs(np.diff(S, axis=1, prepend=S[:, :1]))
        df = np.abs(np.diff(S, axis=0, prepend=S[:1, :]))
        C1 = self._minmax01(dt + df)

        # C2: energía local (suavizado - base) ⇒ resalta picos
        Sm = gaussian_filter(S, sigma=1.0)
        C2 = self._minmax01(np.maximum(Sm - Sm.mean(), 0.0))

        return np.stack([C0, C1, C2], axis=0).astype(np.float32)

    def compute(self, iq_signal: np.ndarray, show: bool = False) -> np.ndarray:
        S = self._stft_logmag(iq_signal)
        if self.mode == "stft":
            if show:
                plt.imshow(S, aspect='auto', origin='lower', cmap='viridis'); plt.title("STFT"); plt.axis("off"); plt.show()
            return S.astype(np.float32)

        if self.mode == "mce":
            M = self._mce_3ch(S)  # [3,H,W]
            if show:
                fig, axs = plt.subplots(1, 3, figsize=(12, 3))
                titles = ["C0: log|S|", "C1: grad", "C2: local energy"]
                for i in range(3):
                    axs[i].imshow(M[i], aspect='auto', origin='lower', cmap='viridis')
                    axs[i].set_title(titles[i]); axs[i].axis("off")
                plt.tight_layout(); plt.show()
            return M

        raise ValueError("mode must be 'stft' or 'mce'")
