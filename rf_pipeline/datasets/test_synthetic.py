# test_synthetic.py — Validación del pipeline con señales reales del TorchSigDataset
# Uso:
#   python test_synthetic.py --savefigs
#   python test_synthetic.py --force_regen --savefigs

import os, glob, argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from scipy.signal import stft
import h5py

# ---- Módulos del pipeline ----
from preprocessing import Preprocessor
from roi_detection import ROIDetector, ROI
from feature_engineering import FeatureEngineer
from visual_stream import VisualStream
from models.hybrid_classifier import HybridRFClassifier

# =========================
# Configuración
# =========================
FS = 10_000_000
NPY_GLOB = "synthetic_signal_*.npy"
GT_CSV = "synthetic_rf_ground_truth_with_bins.csv"
CKPT_DEFAULT = os.path.join("rf_pipeline/checkpoints", "hybrid_from_yolo.pt")

# Detector ajustado a versión estable actual
DETECTOR_KW = dict(
    method="adaptive",
    margin_bins=0,
    gauss_sigma=0.7,
    min_area=10,
    adaptive_window=9,
    adaptive_k=-0.12,
    min_ar=0.3,
    max_ar=8.0,
    min_texture=0.1
)

# =========================
# Auxiliares
# =========================
_rng = np.random.default_rng(42)

def _to_time_bins(s, e, nperseg=32, noverlap=16):
    hop = nperseg - noverlap
    tb_start = int(np.floor(s / hop))
    tb_end = int(np.ceil((e - nperseg) / hop))
    tb_end = max(tb_end, tb_start)
    return tb_start, tb_end


# =========================
# Generación de señales reales
# =========================
def nms_rois(rois, overlap_thresh=0.3, prefer_small=True):
    """
    Aplica Non-Maximum Suppression (NMS) sobre una lista de ROIs.
    Si prefer_small=True, prioriza las ROIs pequeñas cuando hay solapamiento.
    """
    if len(rois) == 0:
        return []

    boxes = np.array([[r.x1, r.y1, r.x2, r.y2] for r in rois])
    scores = np.array([r.score for r in rois])

    # Calcular áreas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Ordenar: pequeñas primero si prefer_small=True
    order = np.argsort(areas if prefer_small else -areas)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(rois[i])
        rest = order[1:]
        new_order = []

        for j in rest:
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = areas[i] + areas[j] - inter
            iou = inter / (union + 1e-12)

            # Si el solapamiento es alto, descartar la ROI grande
            if iou > overlap_thresh:
                if prefer_small and areas[j] > areas[i]:
                    continue  # mantener la pequeña
                elif not prefer_small and areas[j] < areas[i]:
                    continue
            new_order.append(j)

        order = np.array(new_order)

    return keep


def maybe_generate_signals(force=False):
    files = sorted(glob.glob(NPY_GLOB))
    if files and not force and Path(GT_CSV).exists():
        print(f"[OK] Encontradas señales ({len(files)}) y GT existente.")
        return

    print("[LOAD] Extrayendo 3 señales desde TorchSigDataset...")
    FILE = r"C:\Users\User\Documents\GitHub\torchsig\scripts\TorchSigDataset\train\data.h5"
    with h5py.File(FILE, "r") as f:
        keys = [k for k in f.keys() if k.isdigit()]
        selected = random.sample(keys, 3)
        print(f"[INFO] Muestras seleccionadas: {selected}")

        gt_rows = []
        for i, key in enumerate(selected):
            data = np.array(f[f"{key}/data"])
            if np.iscomplexobj(data):
                iq = data
            elif data.ndim == 1:
                iq = data[::2] + 1j * data[1::2]
            elif data.ndim == 2 and data.shape[1] == 2:
                iq = data[:, 0] + 1j * data[:, 1]
            else:
                print(f"[WARN] Formato desconocido: {data.shape}, se omite.")
                continue

            np.save(f"synthetic_signal_{i}.npy", iq)

            meta = f[f"{key}/metadata/0"]
            class_name = meta["class_name"][()].decode("utf-8")
            snr = float(meta["snr_db"][()])
            start = int(meta["start_in_samples"][()])
            dur = int(meta["duration_in_samples"][()])
            f_low = float(meta["_lower_frequency"][()])
            f_high = float(meta["_upper_frequency"][()])

            tb0, tb1 = _to_time_bins(start, start + dur)
            gt_rows.append(dict(
                signal_index=i,
                class_name=class_name,
                snr_db=snr,
                start_sample=start,
                end_sample=start + dur,
                f_low=f_low,
                f_high=f_high,
                tb_start=tb0,
                tb_end=tb1
            ))

    pd.DataFrame(gt_rows).to_csv(GT_CSV, index=False)
    print(f"[OK] Guardado GT en {GT_CSV} y señales synthetic_signal_*.npy")


# =========================
# Evaluación
# =========================
def iou_1d(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / (union + 1e-12)


def evaluate_one(name, iq, gt_df, model, device, savefigs=False, outdir="tests_out", force_gt_rois=False):
    """
    Evalúa una señal con el modelo híbrido.
    Si force_gt_rois=True, los ROIs se generan desde el Ground Truth (GT)
    en lugar de ser detectados automáticamente.
    """
    pre = Preprocessor(fs=FS, nperseg=32, noverlap=16, mode="mce")
    S_mce = pre.compute(iq, show=False)
    S_det = S_mce[1]  # canal de gradiente (detección)

    if force_gt_rois:
        # --- Forzar ROIs desde el CSV de ground truth ---
        rois = []
        H, W = S_det.shape
        for _, r in gt_df.iterrows():
            # Bins de tiempo (x1,x2)
            x1 = int(r.tb_start)
            x2 = int(r.tb_end)
            x1 = max(0, min(W - 1, x1))
            x2 = max(x1 + 1, min(W, x2))

            # Bins de frecuencia (y1,y2)
            f_low = float(r.f_low)
            f_high = float(r.f_high)
            y1 = int((f_low / (FS / 2)) * H)
            y2 = int((f_high / (FS / 2)) * H)
            y1, y2 = sorted((max(0, y1), min(H - 1, y2)))

            roi = ROI(y1=y1, x1=x1, y2=y2, x2=x2, score=1.0, snr_db=float(r.snr_db))
            rois.append(roi)
        print(f"[{name}] Forzando {len(rois)} ROIs desde GT.")
    else:
        # --- Detección normal con ROIDetector ---
        det = ROIDetector(**DETECTOR_KW)
        rois = det.detect(S_det)
        rois = nms_rois(rois, overlap_thresh=0.4)
        print(f"[{name}] Detección automática — {len(rois)} ROIs encontrados.")

    # --- Ingeniería de características ---
    fe = FeatureEngineer(fs=FS, nperseg=32, noverlap=16)
    roi_feats = fe.features_for_rois(iq, S_det, rois)
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

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(S_mce[0], aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f"{name} — {'GT' if force_gt_rois else 'Detección ROI'} (Verde=Señal, Rojo=No-señal)")
    ax.axis("off")

    if not keep:
        print(f"[{name}] No se detectaron ROIs válidos.")
        return dict(sig=name, n_rois=0, tp=0, fp=0, fn=len(gt_df))

    X_eng = torch.stack(X_eng).to(device)
    X_vis = torch.stack(X_vis).to(device)

    # --- Inferencia ---
    model.eval()
    with torch.no_grad():
        logits = model(X_vis, X_eng)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    gt_ranges = [(int(r.tb_start), int(r.tb_end)) for _, r in gt_df.iterrows()]
    tp = fp = 0

    for k, idx in enumerate(keep):
        r = rois[idx]
        p_si = float(probs[k, 1])
        is_sig = p_si >= 0.5
        color = "lime" if is_sig else "red"

        ax.add_patch(mpatches.Rectangle((r.x1, r.y1), r.x2-r.x1, r.y2-r.y1,
                                        linewidth=1.5, edgecolor=color, facecolor='none'))
        ax.text(r.x1, max(0, r.y1-1),
                f"{p_si*100:.1f}% señal" if is_sig else f"{(1-p_si)*100:.1f}% no-señal",
                color=color, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.35))

        best_iou = max((iou_1d(r.x1, r.x2, g0, g1) for g0, g1 in gt_ranges), default=0.0)
        if is_sig and best_iou >= 0.3:
            tp += 1
        elif is_sig:
            fp += 1

    fn = len(gt_ranges) - tp

    if savefigs:
        Path(outdir).mkdir(exist_ok=True)
        fig.savefig(Path(outdir, f"{name}_detections.png"), dpi=150, bbox_inches="tight")

    plt.show()
    print(f"[{name}] ROIs={len(keep)} | TP={tp}  FP={fp}  FN={fn}")
    return dict(sig=name, n_rois=len(keep), tp=tp, fp=fp, fn=fn)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_regen", action="store_true", help="Regenerar señales sintéticas desde TorchSigDataset")
    ap.add_argument("--savefigs", action="store_true", help="Guardar figuras de detección")
    ap.add_argument("--ckpt", type=str, default=CKPT_DEFAULT, help="Ruta del checkpoint del modelo")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force_gt_rois", action="store_true", help="Forzar ROIs desde el ground truth en lugar del detector")
    args = ap.parse_args()

    maybe_generate_signals(force=args.force_regen)
    device = torch.device(args.device)

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.ckpt}")

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

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"[OK] Modelo cargado desde {args.ckpt}")

    files = sorted(glob.glob(NPY_GLOB))
    gt_all = pd.read_csv(GT_CSV)

    summary = []
    for i, p in enumerate(files):
        iq = np.load(p)
        gt_i = gt_all[gt_all["signal_index"] == i].reset_index(drop=True)
        res = evaluate_one(
            os.path.basename(p), iq, gt_i, model, device,
            savefigs=args.savefigs,
            force_gt_rois=args.force_gt_rois
        )
        summary.append(res)

    tp = sum(r["tp"] for r in summary)
    fp = sum(r["fp"] for r in summary)
    fn = sum(r["fn"] for r in summary)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)

    print("\n=== RESUMEN GLOBAL ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  |  Precision={prec:.3f}  Recall={rec:.3f}")


if __name__ == "__main__":
    main()


