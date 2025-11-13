# main.py — Inferencia con MCE (3 canales) y detector estricto

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_loader import IQDataLoader
from preprocessing import Preprocessor
from roi_detection import ROIDetector, ROI
from feature_engineering import FeatureEngineer
from visual_stream import VisualStream
from models.hybrid_classifier import HybridRFClassifier

ZARR_PATH = r"C:\Users\User\Documents\Git\torchsig\dataset\torchsig_narrowband_impaired\data\0000000000.zarr"
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "hybrid_mce_medium_auto.pt")
META_PATH = os.path.join(CKPT_DIR, "metadata.json")

def must_file(path: str, desc: str):
    if not os.path.exists(path):
        print(f"[ERROR] Falta {desc}: {path}")
        print("        Aborting (modo estricto).")
        sys.exit(1)

def key_roi(r: ROI):
    return (r.y1, r.x1, r.y2, r.x2)

if __name__ == "__main__":
    device = torch.device("cpu")

    must_file(META_PATH, "metadata.json")
    must_file(CKPT_PATH, "checkpoint entrenado")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_count = int(meta.get("train_count", 0))
    total_signals = int(meta.get("total_signals", 0))
    train_split = float(meta.get("train_split", 0.7))
    if total_signals <= 0 or train_count <= 0 or train_count >= total_signals:
        print(f"[ERROR] Metadata inconsistente: {meta}")
        sys.exit(1)

    test_index = train_count
    print(f"[INFO] total_signals={total_signals} | train_count={train_count} | test_index={test_index} (primer test no visto)")

    # Step 0 — señal NO vista
    loader = IQDataLoader(ZARR_PATH)
    it = loader.stream_signals()
    iq = None
    for _ in range(test_index + 1):
        try:
            iq = next(it)
        except StopIteration:
            print("[ERROR] El .zarr no tiene suficientes señales para alcanzar el índice de test.")
            sys.exit(1)

    # Step 1 — MCE 3 canales
    pre = Preprocessor(fs=1e6, nperseg=32, noverlap=16, mode="mce")
    S_mce = pre.compute(iq, show=False)  # [3,H,W]

    # Step 2 — Detector estricto sobre C1
    S_det = S_mce[1]
    detector = ROIDetector(
        method="adaptive",
        margin_bins=0,
        gauss_sigma=1,
        min_area=20,
        adaptive_window=9,
        adaptive_k=0
    )
    rois = detector.detect(S_det)
    if len(rois) == 0:
        print("[WARN] No se detectaron ROIs en la señal de TEST. Ajusta parámetros o elige otra señal.")
        sys.exit(0)

    # Step 3A — 32-D engineered
    fe = FeatureEngineer(fs=1e6, nperseg=32, noverlap=16)
    roi_feats = fe.features_for_rois(iq, S_det, rois)
    feat_by_box = {key_roi(rf.roi): rf.features for rf in roi_feats}

    # Step 3B — Visual patches [3,32,32]
    vs = VisualStream(target_size=32)
    patches_list = vs.extract_patches(S_mce, rois, preview=False)

    eng_list, vis_list, keep_indices = [], [], []
    for idx, r in enumerate(rois):
        p = patches_list[idx] if idx < len(patches_list) else None
        f = feat_by_box.get(key_roi(r), None)
        if p is None or f is None:
            continue
        eng_list.append(torch.tensor(f, dtype=torch.float32))
        vis_list.append(torch.tensor(p, dtype=torch.float32))
        keep_indices.append(idx)

    if len(keep_indices) == 0:
        print("[WARN] No hubo coincidencias feat+patch para los ROIs de TEST.")
        sys.exit(0)

    X_eng = torch.stack(eng_list, dim=0).to(device)  # [N,32]
    X_vis = torch.stack(vis_list, dim=0).to(device)  # [N,3,32,32]

    # Step 4 — Modelo (MLP deep + CNN reforzada)
    model = HybridRFClassifier(mlp_variant="deep", cnn_in_channels=3, cnn_embed_dim=64).to(device)
    try:
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"[OK] Checkpoint cargado: {CKPT_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el checkpoint requerido ({CKPT_PATH}).")
        print(f"        Detalle: {e}")
        print("        Aborting (modo estricto).")
        sys.exit(1)

    model.eval()
    with torch.no_grad():
        logits = model(X_eng, X_vis)                   # [N,2]
        probs = F.softmax(logits, dim=1).cpu().numpy() # [N,2]

    # Visualización
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(S_mce[0], aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f"TEST (señal #{test_index}) — Multi-ROI (Verde=Señal, Rojo=No-señal)  |  Split ~{int(train_split*100)}/{int((1-train_split)*100)}")
    ax.axis("off")

    print("\n=== Inferencia Multi-ROI en TEST (no visto) ===")
    for k, idx in enumerate(keep_indices):
        r = rois[idx]
        p_no = float(probs[k, 0])
        p_si = float(probs[k, 1])
        is_sig = p_si >= 0.5
        color = 'lime' if is_sig else 'red'
        label_txt = f"{(p_si*100):.1f}% señal" if is_sig else f"{(p_no*100):.1f}% no-señal"

        rect = mpatches.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1,
                                  linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(r.x1, max(0, r.y1 - 1), label_txt, color=color,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.35))

        decision = "SEÑAL DETECTADA ✅" if is_sig else "NO SEÑAL ❌"
        print(f"ROI {idx:02d}  ({r.y1},{r.x1})→({r.y2},{r.x2})  |  Señal: {p_si*100:.1f}%  |  No-señal: {p_no*100:.1f}%  |  {decision}")

    plt.tight_layout()
    plt.show()
