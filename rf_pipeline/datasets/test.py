# multi_from_torchsig.py — Combina varias señales reales del TorchSigDataset y evalúa detección múltiple
# Uso:
#   python multi_from_torchsig.py --savefigs

import os, random, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py

# ---- Módulos del pipeline ----
from preprocessing import Preprocessor
from roi_detection import ROIDetector
from feature_engineering import FeatureEngineer
from visual_stream import VisualStream
from models.hybrid_classifier import HybridRFClassifier

# =========================
# Configuración
# =========================
FS = 10_000_000
H5_PATH = r"C:\Users\User\Documents\GitHub\torchsig\scripts\TorchSigDataset\train\data.h5"
CKPT_DEFAULT = os.path.join("checkpoints", "hybrid_h5_best.pt")

DETECTOR_KW = dict(
    method="percentile",
    margin_bins=0,
    gauss_sigma=0.7,       # + suavizado → menos ruido
    min_area=5,            # ↓ área mínima → más ROIs pequeñas
    adaptive_window=15,    # ↑ ventana → umbral más estable
    adaptive_k=-0.05,      # ↓ valor → más sensible
    min_ar=0.2,
    max_ar=10.0,
    min_texture=0.04      # ↓ textura mínima → detecta señales débiles
)

def nms_rois(rois, overlap_thresh=0.3):
    """
    Non-Maximum Suppression inverso: conserva ROIs pequeñas cuando se solapan.
    Si una ROI grande cubre una pequeña con IoU > overlap_thresh, se descarta la grande.
    """
    if len(rois) == 0:
        return []

    boxes = np.array([[r.x1, r.y1, r.x2, r.y2] for r in rois])
    scores = np.array([r.score for r in rois])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    keep = np.ones(len(rois), dtype=bool)

    for i in range(len(rois)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(rois)):
            if not keep[j]:
                continue
            # Calcular intersección
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            union = areas[i] + areas[j] - inter
            iou = inter / (union + 1e-12)

            # Si se solapan fuertemente, eliminar la ROI más grande
            if iou > overlap_thresh:
                if areas[i] > areas[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    filtered = [r for k, r in enumerate(rois) if keep[k]]
    return filtered


# =========================
# Utilidad
# =========================
def load_random_signals(h5_path, k=3):
    """Carga k señales aleatorias del TorchSigDataset (data.h5)"""
    with h5py.File(h5_path, "r") as f:
        keys = [k for k in f.keys() if k.isdigit()]
        selected = random.sample(keys, k)
        signals = []
        for key in selected:
            data = np.array(f[f"{key}/data"])
            if np.iscomplexobj(data):
                iq = data
            elif data.ndim == 1:
                iq = data[::2] + 1j * data[1::2]
            elif data.ndim == 2 and data.shape[1] == 2:
                iq = data[:, 0] + 1j * data[:, 1]
            else:
                continue
            signals.append(iq)
    return signals

# =========================
# Prueba de mezcla
# =========================
def multi_signal_test(model, device, savefigs=False):
    """Extrae entre 2 y 5 señales del TorchSigDataset y prueba detección múltiple"""
    if not Path(H5_PATH).exists():
        raise FileNotFoundError(f"No se encontró el dataset en {H5_PATH}")

    k = random.randint(2, 5)
    sigs = load_random_signals(H5_PATH, k=k)
    print(f"[INFO] Extraídas {k} señales del TorchSigDataset")

    # --- Normalización y mezcla ---
    sigs = [s / (np.max(np.abs(s)) + 1e-9) for s in sigs]
    spacing = int(0.15 * max(len(s) for s in sigs))
    total_len = sum(len(s) + spacing for s in sigs)
    mixed = np.zeros(total_len, dtype=np.complex64)

    pos = 0
    for i, s in enumerate(sigs):
        mixed[pos:pos + len(s)] += s
        pos += len(s) + spacing

    # --- Espectrograma ---
    pre = Preprocessor(fs=FS, nperseg=32, noverlap=16, mode="mce")
    S_mce = pre.compute(mixed, show=False)
    S_det = S_mce[1]

    # --- Detección ROI ---
    det = ROIDetector(**DETECTOR_KW)
    rois = det.detect(S_det)
    rois = nms_rois(rois, overlap_thresh=0.01)
    print(f"[DETECT] {len(rois)} ROIs detectadas en mezcla de {k} señales.")

    if len(rois) == 0:
        print("⚠️ No se detectaron ROIs. Verifica los parámetros del detector.")
        return

    # --- Extracción de características ---
    fe = FeatureEngineer(fs=FS, nperseg=32, noverlap=16)
    roi_feats = fe.features_for_rois(mixed, S_det, rois)
    feat_by_box = {(r.roi.y1, r.roi.x1, r.roi.y2, r.roi.x2): r.features for r in roi_feats}

    # --- Parches visuales ---
    vs = VisualStream(target_size=32)
    patches = vs.extract_patches(S_mce, rois, preview=False)

    X_eng, X_vis, keep = [], [], []
    for idx, r in enumerate(rois):
        key = (r.y1, r.x1, r.y2, r.x2)
        if key not in feat_by_box or idx >= len(patches):
            continue
        X_eng.append(torch.tensor(feat_by_box[key], dtype=torch.float32))
        X_vis.append(torch.tensor(patches[idx], dtype=torch.float32))
        keep.append(idx)

    if not keep:
        print("⚠️ No se detectaron ROIs válidas.")
        return

    X_eng = torch.stack(X_eng).to(device)
    X_vis = torch.stack(X_vis).to(device)

    # --- Inferencia ---
    model.eval()
    with torch.no_grad():
        logits = model(X_vis, X_eng)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # --- Visualización ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(S_mce[0], aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f"Detección múltiple ({k} señales reales unidas) — Verde=Señal, Rojo=No-señal")
    ax.axis("off")

    for i, idx in enumerate(keep):
        r = rois[idx]
        p_si = float(probs[i, 1])
        color = "lime" if p_si >= 0.5 else "red"
        ax.add_patch(plt.Rectangle((r.x1, r.y1), r.x2-r.x1, r.y2-r.y1,
                                   edgecolor=color, facecolor='none', lw=1.5))
        ax.text(r.x1, max(0, r.y1-2), f"{p_si*100:.1f}%", color=color,
                fontsize=7, bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))

    plt.tight_layout()
    if savefigs:
        Path("tests_out").mkdir(exist_ok=True)
        plt.savefig("tests_out/multi_signal_test.png", dpi=150, bbox_inches="tight")
    plt.show()


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--savefigs", action="store_true", help="Guardar figura final")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(CKPT_DEFAULT).exists():
        raise FileNotFoundError(f"No se encontró el modelo en {CKPT_DEFAULT}")

    model = HybridRFClassifier(
        num_classes=2,
        feat_dim=32,
        img_in_ch=3,
        cnn_channels=(16, 32, 64),
        cnn_dropout=0.05,
        mlp_hidden=(64, 64),
        mlp_dropout=0.1,
        fusion_hidden=(128,),
        fusion_dropout=0.1
    ).to(device)

    state = torch.load(CKPT_DEFAULT, map_location=device)
    model.load_state_dict(state)
    print(f"[OK] Modelo cargado desde {CKPT_DEFAULT}")

    multi_signal_test(model, device, savefigs=args.savefigs)


if __name__ == "__main__":
    main()
