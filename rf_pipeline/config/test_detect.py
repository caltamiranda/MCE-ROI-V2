import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # <--- CORRECCION: Renombrado para evitar conflicto
import h5py
import random
import os
import sys
import torch.nn.functional as F

# --- Setup de Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Imports locales
import config as cfg
from preprocessing import Preprocessor
from models.hybrid_classifier import HybridRFClassifier
from roi_detection import ROIDetector
from feature_engineering import FeatureEngineer
from visual_stream import VisualStream

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Asegúrate de que apunta al archivo de TEST
TEST_H5_PATH = cfg.ROOT_DIR + r"\test\data.h5" 
MODEL_PATH = "best_hybrid_detector.pt"  # El modelo que acabas de entrenar
NUM_SAMPLES = 10  # Cuántas imágenes quieres ver

# ==========================================
# UTILS
# ==========================================
def meta_to_box(meta_group, S_shape, fs):
    """Convierte metadata a caja (usando lógica fftshift)"""
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

def run_test():
    if not os.path.exists(TEST_H5_PATH):
        print(f"[ERROR] No existe: {TEST_H5_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No existe el modelo: {MODEL_PATH}. ¿Ya entrenaste?")
        return

    print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
    device = cfg.DEVICE
    
    # 1. Cargar Modelo
    model = HybridRFClassifier(
        num_classes=2, feat_dim=32, img_in_ch=3, 
        cnn_channels=(16, 32, 64), mlp_hidden=(64, 64)
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"[ERROR] Falló al cargar pesos: {e}")
        return
        
    model.eval()

    # 2. Inicializar Pipeline (CON LA CONFIGURACIÓN OPTIMIZADA)
    pre = Preprocessor(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP, mode="mce")
    
    detector = ROIDetector(
        method="zscore",
        z_thresh=4,           
        background_window=65,
        gauss_sigma=(0.0, 3.0), # Configuración asimétrica optimizada
        merge_dist=3,          
        min_area=60,           
        min_ar=0.3,
        max_ar=20.0,
    )
    
    fe = FeatureEngineer(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP)
    vs = VisualStream(target_size=cfg.IMG_SIZE)

    print(f"[INFO] Probando con {NUM_SAMPLES} señales de TEST...")
    
    with h5py.File(TEST_H5_PATH, 'r') as f:
        keys = [k for k in f.keys() if k.isdigit()]
        if not keys:
            print("No se encontraron keys en el archivo H5.")
            return
        selected_keys = random.sample(keys, min(len(keys), NUM_SAMPLES))

        for key in selected_keys:
            print(f"--- Procesando ID: {key} ---")
            
            # A. Cargar Señal
            ds = f[key]['data'][:]
            if ds.ndim > 1 and ds.shape[-1] == 2: iq = ds[...,0]+1j*ds[...,1]
            else: iq = ds
            iq = np.squeeze(iq)

            # B. Preprocessing
            S_mce = pre.compute(iq)
            S_mce = np.fft.fftshift(S_mce, axes=1) # CRÍTICO
            S_det = S_mce[1]

            # C. Ground Truth (Solo para visualización)
            gt_boxes = []
            if 'metadata' in f[key]:
                mg = f[key]['metadata']
                for k in mg.keys():
                    b = meta_to_box(mg[k], S_mce.shape, cfg.FS)
                    if b: gt_boxes.append(b)

            # ============================================
            # INFERENCIA REAL
            # ============================================
            
            # 1. Proposal
            rois = detector.detect(S_det)
            
            final_predictions = [] # (box, score)

            if len(rois) > 0:
                # 2. Feature Extraction
                roi_feats = fe.features_for_rois(iq, S_det, rois)
                feat_map = {(r.roi.y1, r.roi.x1, r.roi.y2, r.roi.x2): r.features for r in roi_feats}
                patches = vs.extract_patches(S_mce, rois)

                # 3. Batching
                X_vis, X_eng, coords = [], [], []
                for k, r in enumerate(rois):
                    key_coord = (r.y1, r.x1, r.y2, r.x2)
                    if key_coord in feat_map and k < len(patches) and patches[k] is not None:
                        X_eng.append(torch.tensor(feat_map[key_coord], dtype=torch.float32))
                        X_vis.append(torch.tensor(patches[k], dtype=torch.float32))
                        coords.append(r)

                if len(X_vis) > 0:
                    X_vis = torch.stack(X_vis).to(device)
                    X_eng = torch.stack(X_eng).to(device)

                    # 4. Clasificación
                    with torch.no_grad():
                        logits = model(X_vis, X_eng)
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

                    # 5. Filtrado (Umbral de decisión 0.5)
                    for i, p in enumerate(probs):
                        if p > 0.5: # Solo guardamos si el modelo cree que es señal
                            final_predictions.append((coords[i], p))

            # ============================================
            # VISUALIZACIÓN
            # ============================================
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(S_mce[0], aspect='auto', origin='lower', cmap='viridis')
            
            # Dibujar GT (Verde)
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y2, "GT", color='lime', fontsize=8, verticalalignment='bottom')

            # Dibujar Predicciones (Rojo)
            for (r, score) in final_predictions:
                rect = mpatches.Rectangle((r.x1, r.y1), r.x2-r.x1, r.y2-r.y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(r.x1, r.y1, f"{score:.0%}", color='white', fontsize=8, fontweight='bold', backgroundcolor='red')

            plt.title(f"TEST ID {key} | Verde=Real, Rojo=Predicción (Modelo)")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    run_test()