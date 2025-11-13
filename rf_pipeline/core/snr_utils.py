import numpy as np
from scipy.ndimage import binary_erosion

def roi_snr_db(S, roi, guard_frac=0.5, erode=1):
    """
    Estimate ROI SNR in dB using eroded interior as signal and frequency-guard bands as noise.

    Parameters
    ----------
    S : np.ndarray (H, W)
        Spectrogram-like image (non-negative float).
    roi : object with attributes (y1, x1, y2, x2)
        ROI box in pixel coordinates (row/col).
    guard_frac : float
        Fraction of the ROI height (along frequency axis) used for guard bands (split half up/down).
    erode : int
        Iterations for binary erosion to avoid border contamination in signal estimate.

    Returns
    -------
    float
        Estimated SNR in dB. Returns -inf if estimation is not possible.
    """
    H, W = S.shape
    y1, x1, y2, x2 = int(roi.y1), int(roi.x1), int(roi.y2), int(roi.x2)
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    if y2 <= y1 or x2 <= x1:
        return float('-inf')

    h = y2 - y1
    # Signal region (eroded)
    sig = S[y1:y2, x1:x2]
    mask = np.ones(sig.shape, dtype=bool)
    if erode > 0 and sig.size > 0:
        try:
            mask = binary_erosion(mask, iterations=int(erode), border_value=1)
        except Exception:
            pass
    sig_vals = sig[mask]
    if sig_vals.size == 0:
        return float('-inf')

    # Guard bands above and below along frequency axis
    g = max(1, int(h * guard_frac * 0.5))
    top0, top1 = max(0, y1 - g), y1
    bot0, bot1 = y2, min(H, y2 + g)

    guards = []
    if top1 > top0:
        guards.append(S[top0:top1, x1:x2].ravel())
    if bot1 > bot0:
        guards.append(S[bot0:bot1, x1:x2].ravel())
    if not guards:
        return float('-inf')

    noise = np.concatenate(guards)
    n = noise.size
    if n == 0:
        return float('-inf')

    # Trimmed mean (10%) for robustness
    if n > 20:
        noise = np.sort(noise)
        k = max(1, n // 10)
        noise = noise[k:-k]

    Pn = float(np.mean(noise**2) + 1e-12)
    Ps = float(np.mean(sig_vals**2) + 1e-12)
    snr_lin = max(Ps - Pn, 1e-12) / Pn
    return 10.0 * np.log10(snr_lin)
