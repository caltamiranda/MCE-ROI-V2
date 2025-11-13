#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Genera un dataset unificado de ROIs (señal / no-señal) desde rf_dataset/ (make_rf_dataset.py).
- Recorre train + val + test (y todas las clases).
- Para cada .npy: MCE[3ch] -> detección ROI (C1) -> features 32-D -> parche 3x32x32.
- Crea negativos realistas no solapados y balancea ~50/50.
- Guarda: roi_dataset.pt con {"features": [N,32], "patches": [N,3,32,32], "labels": [N]}.

Uso:
  python generate_roi_dataset.py --root "C:\\Users\\User\\Documents\\Git\\MCE-ROI\\rf_dataset" --out roi_dataset.pt
"""

import os
import sys
import glob
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import cv2  # para redimensionado de negativos consistentes

# --- Importa tus módulos del pipeline (mismo repo/proyecto) ---
# Si este script vive en la raíz del repo, esto debería funcionar tal cual.
# Si no, añade el path del proyecto:
# sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import Preprocessor            # :contentReference[oaicite:4]{index=4}
from roi_detection import ROIDetector, ROI        # :contentReference[oaicite:5]{index=5}
from feature_engineering import FeatureEngineer   # :contentReference[oaicite:6]{index=6}
from visual_stream import VisualStream            # :contentReference[oaicite:7]{index=7}


# -------------------------------
# Config por defecto (coherente con tu pipeline)
# -------------------------------
FS = 1_000_000
NPERSEG = 32
NOVERLAP = 16
MCE_MODE = "mce"

# Detector "estricto" como en entrenamiento base (train.py)
DETECTOR_KW = dict(
    method="adaptive",
    margin_bins=0,
    gauss_sigma=0.6,
    min_area=20,
    adaptive_window=9,
    adaptive_k=0.0,
)

TARGET_SIZE = 32            # parches 32x32
NEG_PER_SIGNAL = 4          # intentos de negativos por señal (tope)
MIN_NEG_AREA = 20           # igual que train.py
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _list_npy(root: str) -> List[str]:
    """Encuentra todos los .npy bajo train/ val/ test/ (todas las clases)."""
    patt = [
        os.path.join(root, "train", "*", "*.npy"),
        os.path.join(root, "val", "*", "*.npy"),
        os.path.join(root, "test", "*", "*.npy"),
    ]
    files = []
    for p in patt:
        files.extend(glob.glob(p))
    files = sorted(files)
    return files


def _to_complex(iq2: np.ndarray) -> np.ndarray:
    """
    Convierte [N,2] (I,Q float32) -> complejo 1D.
    """
    if iq2.ndim != 2 or iq2.shape[1] != 2:
        raise ValueError(f"Esperaba array [N,2] (I,Q). Llegó {iq2.shape}")
    return iq2[:, 0].astype(np.float32) + 1j * iq2[:, 1].astype(np.float32)


def _non_overlapping_random_roi(H: int, W: int, rois: List[ROI], min_area=20, max_tries=60) -> Optional[ROI]:
    """
    Genera un ROI negativo rectangular que no se solape con ningún ROI positivo.
    Lógica equivalente a la usada en train.py para negativos.  (balance realista)
    """
    def overlaps(r: ROI, s: ROI) -> bool:
        return not (r.x2 <= s.x1 or r.x1 >= s.x2 or r.y2 <= s.y1 or r.y1 >= s.y2)

    for _ in range(max_tries):
        h = random.randint(4, max(4, H // 6))
        w = random.randint(4, max(4, W // 6))
        y1 = random.randint(0, max(0, H - h))
        x1 = random.randint(0, max(0, W - w))
        y2, x2 = y1 + h, x1 + w
        cand = ROI(y1=y1, x1=x1, y2=y2, x2=x2, score=0.0)
        if (h * w) < min_area:
            continue
        if any(overlaps(cand, r) for r in rois):
            continue
        return cand
    return None


def _generate_negatives(S_mce: np.ndarray, S_det: np.ndarray, pos_rois: List[ROI],
                        max_negatives: int) -> List[np.ndarray]:
    """
    Genera parches negativos realistas desde C0 (log|S|), sin solaparse con ROIs positivos,
    descartando zonas brillantes que suelen corresponder a señal.
    Devuelve lista de parches [3,32,32] listos (normalizados).
    """
    C0 = S_mce[0]    # canal base log|S|
    H, W = S_det.shape
    patches = []

    # Umbral de brillo para evitar falsos negativos "engañosos"
    bright_thr = float(np.percentile(C0, 70))
    vs = VisualStream(target_size=TARGET_SIZE)   # normaliza como en tu pipeline  :contentReference[oaicite:8]{index=8}

    tries = 0
    while len(patches) < max_negatives and tries < max_negatives * 6:
        tries += 1
        rneg = _non_overlapping_random_roi(H, W, pos_rois, min_area=MIN_NEG_AREA)
        if rneg is None:
            continue

        patch_b = C0[rneg.y1:rneg.y2, rneg.x1:rneg.x2]
        if patch_b.size == 0:
            continue

        # Evita regiones demasiado brillantes (probable señal)
        if float(np.mean(patch_b)) > bright_thr:
            continue

        # Redimensiona y normaliza a 3 canales replicados
        pr = cv2.resize(patch_b, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        pr = np.log1p(pr)
        pr = (pr - pr.min()) / (pr.ptp() + 1e-12)
        S_neg = np.stack([pr, pr, pr], axis=0).astype(np.float32)   # [3,32,32]
        patches.append(S_neg)

    return patches


def build_roi_dataset(root: str, out_path: str, max_files: Optional[int] = None) -> None:
    """
    Recorre rf_dataset/ y construye el dataset .pt unificado (balanceado ~50/50).
    """
    files = _list_npy(root)
    if not files:
        raise FileNotFoundError(f"No encontré .npy en {root}. ¿Ejecutaste make_rf_dataset.py?")

    if max_files is not None:
        files = files[:max_files]

    print(f"[INFO] Encontrados {len(files)} archivos .npy en: {root}")

    pre = Preprocessor(fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, mode=MCE_MODE)     # :contentReference[oaicite:9]{index=9}
    det = ROIDetector(**DETECTOR_KW)                                                 # :contentReference[oaicite:10]{index=10}
    fe = FeatureEngineer(fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)                  # :contentReference[oaicite:11]{index=11}
    vs = VisualStream(target_size=TARGET_SIZE)                                       # :contentReference[oaicite:12]{index=12}

    feats_pos, patches_pos = [], []
    feats_neg, patches_neg = [], []
    n_files_ok, n_files_skip = 0, 0
    total_pos_rois = 0

    for i, p in enumerate(files, start=1):
        try:
            iq2 = np.load(p)
            iq = _to_complex(iq2)
        except Exception as e:
            print(f"[WARN] Saltando {p}: no es [N,2] válido ({e})")
            n_files_skip += 1
            continue

        # Step 1 — MCE 3 canales
        S_mce = pre.compute(iq, show=False)        # [3,H,W]
        S_det = S_mce[1]                           # detector trabaja sobre C1 (grad/edges) (coherente con main.py)

        # Step 2 — ROIs positivos
        rois = det.detect(S_det)
        if not rois:
            n_files_ok += 1
            continue

        # Step 3A — features 32-D
        roi_feats = fe.features_for_rois(iq, S_det, rois)
        def key_roi(r): return (r.y1, r.x1, r.y2, r.x2)
        feat_by_box = {key_roi(rf.roi): rf.features for rf in roi_feats}

        # Step 3B — parches 3×32×32 (MCE)
        patches = vs.extract_patches(S_mce, rois, preview=False)

        # Append positivos válidos
        for j, r in enumerate(rois):
            if j >= len(patches): break
            patch = patches[j]
            if patch is None or patch.size == 0:
                continue
            f = feat_by_box.get(key_roi(r), None)
            if f is None:
                continue
            feats_pos.append(f.astype(np.float32))
            patches_pos.append(patch.astype(np.float32))

        total_pos_rois += len(patches_pos)  # acumulado parcial

        # Negativos realistas (replicando lógica de train.py)
        neg_patches = _generate_negatives(S_mce, S_det, rois, max_negatives=NEG_PER_SIGNAL)
        # Para negativos, usamos features nulos (32) por coherencia con train.py
        for pn in neg_patches:
            feats_neg.append(np.zeros(32, dtype=np.float32))
            patches_neg.append(pn.astype(np.float32))

        n_files_ok += 1
        if i % 50 == 0:
            print(f"[INFO] Procesados {i}/{len(files)} archivos…  pos={len(feats_pos)}  neg={len(feats_neg)}")

    # -----------------------
    # Balanceo 50/50
    # -----------------------
    n_pos = len(feats_pos)
    n_neg = len(feats_neg)
    if n_pos == 0:
        raise RuntimeError("No se obtuvieron ROIs positivos. Ajusta parámetros del detector o revisa las señales.")

    # Empareja al mínimo
    n_keep = min(n_pos, n_neg)
    if n_keep == 0:
        # Si no hubo negativos, generamos algunos desde todos los espectros (muy raro)
        print("[WARN] No hay negativos suficientes; el dataset quedará solo con positivos (no recomendado).")
        n_keep = n_pos

    # Submuestreo aleatorio para balancear
    idx_pos = np.random.permutation(n_pos)[:n_keep]
    idx_neg = np.random.permutation(n_neg)[:n_keep]

    feats_bal = np.concatenate([np.stack([feats_pos[k] for k in idx_pos], axis=0),
                                np.stack([feats_neg[k] for k in idx_neg], axis=0)], axis=0)
    patches_bal = np.concatenate([np.stack([patches_pos[k] for k in idx_pos], axis=0),
                                  np.stack([patches_neg[k] for k in idx_neg], axis=0)], axis=0)
    labels_bal = np.concatenate([np.ones(n_keep, dtype=np.int64),
                                 np.zeros(n_keep, dtype=np.int64)], axis=0)

    # Baraja final (mezcla pos/neg)
    perm = np.random.permutation(len(labels_bal))
    feats_bal = feats_bal[perm]
    patches_bal = patches_bal[perm]
    labels_bal = labels_bal[perm]

    # -----------------------
    # Guardado
    # -----------------------
    out = {
        "features": torch.tensor(feats_bal, dtype=torch.float32),        # [N,32]
        "patches":  torch.tensor(patches_bal, dtype=torch.float32),      # [N,3,32,32]
        "labels":   torch.tensor(labels_bal, dtype=torch.long),          # [N]
        "meta": {
            "fs": FS,
            "nperseg": NPERSEG,
            "noverlap": NOVERLAP,
            "mce_mode": True,
            "detector": DETECTOR_KW,
            "source_root": str(root),
            "files_seen": len(files),
            "files_ok": n_files_ok,
            "files_skipped": n_files_skip,
        }
    }
    torch.save(out, out_path)

    print("\n====================")
    print("      RESUMEN       ")
    print("====================")
    print(f"[DATASET] ROIs totales (balanceados): {len(labels_bal)} → "
          f"{int(labels_bal.sum())} señal / {int((1-labels_bal).sum())} no-señal")
    print(f"[INFO] Guardado en: {out_path}")
    print(f"[SRC]  archivos .npy: {len(files)} (ok={n_files_ok}, skip={n_files_skip})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Ruta a rf_dataset (output de make_rf_dataset.py)")
    ap.add_argument("--out", type=str, default="roi_dataset.pt",
                    help="Ruta de salida .pt (PyTorch)")
    ap.add_argument("--max_files", type=int, default=None,
                    help="(Opcional) Límite de .npy a procesar para pruebas rápidas")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_path = os.path.abspath(args.out)
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    print(f"[CFG] root={root}")
    print(f"[CFG] out={out_path}")
    if args.max_files:
        print(f"[CFG] max_files={args.max_files}")

    build_roi_dataset(root=root, out_path=out_path, max_files=args.max_files)


if __name__ == "__main__":
    main()
