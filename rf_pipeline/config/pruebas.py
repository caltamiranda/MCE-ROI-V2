import h5py
import numpy as np
import random
import sys
import os
import time
from copy import deepcopy

# --- Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config as cfg
from preprocessing import Preprocessor
from roi_detection import ROIDetector

# ==========================================
# CONFIGURACIÓN DE BÚSQUEDA (Z-SCORE)
# ==========================================
NUM_SAMPLES = 25       # Usamos 25 señales para tener variedad
NUM_TRIALS = 500       # Pruebas rápidas
H5_PATH = cfg.TRAIN_H5

# Espacio de búsqueda específico para el método estadístico
PARAM_SPACE = {
    "z_thresh": (1.5, 5.0),         # <--- El más importante. 2.0 es sensible, 4.0 es estricto
    "background_window": [25, 35, 45, 55, 65], # Tamaño para calcular el piso de ruido
    "gauss_sigma": (0.5, 2.5),      # Suavizado previo
    "merge_dist": [1, 2, 3, 4, 5],  # Distancia para unir fragmentos
    "min_area": [5, 8, 12, 15],     # Tamaño mínimo
    "min_ar": [0.1],                # Fijo (permitir señales delgadas)
    "max_ar": [20.0]                # Fijo
}

# ==========================================
# UTILIDADES (Igual que antes)
# ==========================================
def meta_to_box(meta_group, S_shape, fs):
    C, H, W = S_shape
    try:
        start = meta_group['start_in_samples'][()]
        dur = meta_group['duration_in_samples'][()]
        if '_lower_frequency' in meta_group: flo = meta_group['_lower_frequency'][()]
        elif 'low_freq' in meta_group: flo = meta_group['low_freq'][()]
        else: return None
        if '_upper_frequency' in meta_group: fhi = meta_group['_upper_frequency'][()]
        elif 'high_freq' in meta_group: fhi = meta_group['high_freq'][()]
        else: return None
    except: return None

    hop = cfg.NPERSEG - cfg.NOVERLAP
    x1, x2 = int(start/hop), int((start+dur)/hop)
    f_min, f_max = -fs/2, fs/2
    y1 = int(((flo - f_min)/(f_max - f_min)) * H)
    y2 = int(((fhi - f_min)/(f_max - f_min)) * H)
    
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    y1, y2 = sorted([y1, y2])
    if x2<=x1: x2=x1+1
    if y2<=y1: y2=y1+1
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-12)

def load_cache():
    print(f"[INFO] Cargando {NUM_SAMPLES} muestras en RAM...")
    pre = Preprocessor(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP, mode="mce")
    cache = []
    with h5py.File(H5_PATH, 'r') as f:
        keys = [k for k in f.keys() if k.isdigit()]
        # Mezclar para tener variedad
        random.shuffle(keys)
        
        count = 0
        for key in keys:
            if count >= NUM_SAMPLES: break
            
            # Cargar
            ds = f[key]['data'][:]
            if ds.ndim > 1 and ds.shape[-1] == 2: iq = ds[...,0]+1j*ds[...,1]
            else: iq = ds
            iq = np.squeeze(iq)
            
            # Procesar
            S_mce = pre.compute(iq)
            S_mce = np.fft.fftshift(S_mce, axes=1)
            S_det = S_mce[1] # Gradiente
            
            # GT
            gt = []
            if 'metadata' in f[key]:
                mg = f[key]['metadata']
                for k in mg.keys():
                    b = meta_to_box(mg[k], S_mce.shape, cfg.FS)
                    if b: gt.append(b)
            
            if len(gt) > 0:
                cache.append({"S_det": S_det, "gt": gt})
                count += 1
    print(f"[OK] {len(cache)} muestras cargadas.")
    return cache

def get_random_params():
    p = {}
    p["method"] = "zscore" # Forzamos el nuevo método
    p["z_thresh"] = round(random.uniform(*PARAM_SPACE["z_thresh"]), 2)
    p["background_window"] = random.choice(PARAM_SPACE["background_window"])
    p["gauss_sigma"] = round(random.uniform(*PARAM_SPACE["gauss_sigma"]), 2)
    p["merge_dist"] = random.choice(PARAM_SPACE["merge_dist"])
    p["min_area"] = random.choice(PARAM_SPACE["min_area"])
    p["min_ar"] = 0.1
    p["max_ar"] = 20.0
    return p

def evaluate(params, cache):
    detector = ROIDetector(**params)
    tp, fp, fn = 0, 0, 0
    
    for item in cache:
        S_det = item["S_det"]
        gts = item["gt"]
        
        rois = detector.detect(S_det)
        preds = [[r.x1, r.y1, r.x2, r.y2] for r in rois]
        
        matched_gt = [False]*len(gts)
        
        for p in preds:
            hit = False
            for i, g in enumerate(gts):
                if calculate_iou(p, g) > 0.2: # IoU suave para 'detection'
                    matched_gt[i] = True
                    hit = True
            if hit: tp += 1
            else: fp += 1
            
        fn += sum(1 for m in matched_gt if not m)
        
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2*(prec*rec)/(prec+rec+1e-12)
    return f1, rec, prec

# ==========================================
# MAIN
# ==========================================
def main():
    cache = load_cache()
    if not cache: return
    
    best_f1 = -1.0
    best_recall = -1.0
    best_p = None
    
    print(f"\n[START] Optimizando Z-Score ({NUM_TRIALS} trials)...")
    print(f"{'F1':<7} | {'Rec':<7} | {'Prec':<7} | Z-Thresh | Win | Sigma")
    print("-" * 60)
    
    t0 = time.time()
    for i in range(NUM_TRIALS):
        p = get_random_params()
        f1, rec, prec = evaluate(p, cache)
        
        # Guardar el mejor
        # CRITERIO: Queremos F1 alto, pero PENALIZAMOS FUERTE si el Recall baja de 0.85
        # (Preferimos ruido a perder señal)
        score = f1
        if rec < 0.80: score = score * 0.5 # Penalización
        
        if score > best_f1:
            best_f1 = score
            best_recall = rec
            best_p = deepcopy(p)
            print(f"{f1:.4f}  | {rec:.4f}  | {prec:.4f}  | {p['z_thresh']:<8} | {p['background_window']:<3} | {p['gauss_sigma']}")
            
    print("-" * 60)
    print(f"Tiempo: {time.time()-t0:.1f}s")
    print(f"MEJOR RESULTADO -> F1: {best_f1:.4f} (Recall: {best_recall:.4f})")
    
    print("\n=== COPIAR EN TRAIN_DETECTION_HYBRID.PY ===")
    print("detector = ROIDetector(")
    for k,v in best_p.items():
        print(f"    {k}={repr(v)},")
    print(")")

if __name__ == "__main__":
    main()