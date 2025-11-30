# rf_pipeline/h5_dataloader.py
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import os
import sys

# --- Setup de Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from core.preprocessing import Preprocessor
from core.feature_engineering import FeatureEngineer
from core.visual_stream import VisualStream
from core.roi_detection import ROI
import config as cfg

class H5HybridDetectionDataset(Dataset):
    def __init__(self, h5_path, mode="train"):
        self.h5_path = h5_path
        self.mode = mode
        self.keys = []
        
        # Leer keys disponibles
        with h5py.File(h5_path, 'r') as f:
            self.keys = [k for k in f.keys() if k.isdigit()]
            
        # Pipeline
        self.pre = Preprocessor(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP, mode="mce")
        self.fe = FeatureEngineer(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP)
        self.vs = VisualStream(target_size=cfg.IMG_SIZE)

    def __len__(self):
        return len(self.keys)

    def _meta_to_box(self, meta_group, S_shape):
        """Lógica validada en pruebas.py"""
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

        # Tiempo
        hop = cfg.NPERSEG - cfg.NOVERLAP
        x1 = int(start / hop)
        x2 = int((start + dur) / hop)
        
        # Frecuencia (Mapeo [-Fs/2, Fs/2] -> [0, H])
        f_min, f_max = -cfg.FS/2, cfg.FS/2
        y1 = int(((flo - f_min)/(f_max - f_min)) * H)
        y2 = int(((fhi - f_min)/(f_max - f_min)) * H)

        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        y1, y2 = sorted([y1, y2])
        
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1
        return [x1, y1, x2, y2]

    def _get_negative_box(self, S_shape, gt_boxes):
        C, H, W = S_shape
        for _ in range(10):
            h = random.randint(5, H//2)
            w = random.randint(10, W//4)
            y1 = random.randint(0, max(0, H-h))
            x1 = random.randint(0, max(0, W-w))
            box = [x1, y1, x1+w, y1+h]
            
            # Chequear intersección simple
            intersect = False
            for gt in gt_boxes:
                if not (box[2] <= gt[0] or box[0] >= gt[2] or box[3] <= gt[1] or box[1] >= gt[3]):
                    intersect = True; break
            if not intersect: return box
        return [0, 0, 32, 32] # Fallback

    def __getitem__(self, idx):
        # Abrir archivo en cada llamada (seguro para multithreading)
        with h5py.File(self.h5_path, 'r') as f:
            key = self.keys[idx]
            
            # 1. Cargar IQ y Aplanar
            ds = f[key]['data'][:]
            if ds.ndim > 1 and ds.shape[-1] == 2:
                iq = ds[..., 0] + 1j * ds[..., 1]
            else: iq = ds
            iq = np.squeeze(iq) # [N]

            # 2. Espectrograma + FFTSHIFT
            S_mce = self.pre.compute(iq)
            S_mce = np.fft.fftshift(S_mce, axes=1) # CRÍTICO: Alinear freq negativa/positiva
            S_det = S_mce[1] # Canal gradiente

            # 3. Obtener GT Boxes
            gt_boxes = []
            if 'metadata' in f[key]:
                m_grp = f[key]['metadata']
                for k in m_grp.keys():
                    b = self._meta_to_box(m_grp[k], S_mce.shape)
                    if b: gt_boxes.append(b)

        # --- MODO EVAL (Full Scene) ---
        if self.mode == "eval":
            return {
                "S_mce": S_mce, "S_det": S_det, "iq": iq,
                "gt_boxes": np.array(gt_boxes, dtype=np.float32)
            }

        # --- MODO TRAIN (Crops Híbridos) ---
        # 50% Señal / 50% Ruido
        label = 1.0 if (gt_boxes and random.random() > 0.5) else 0.0
        target_box = random.choice(gt_boxes) if label == 1.0 else self._get_negative_box(S_mce.shape, gt_boxes)
        
        x1, y1, x2, y2 = target_box
        roi_obj = ROI(y1=y1, x1=x1, y2=y2, x2=x2, score=0.0)

        # A. MLP Features (Estadísticas IQ)
        feats = self.fe.features_for_rois(iq, S_det, [roi_obj])
        f_vec = torch.tensor(feats[0].features, dtype=torch.float32) if feats else torch.zeros(32)

        # B. CNN Features (Imagen)
        patches = self.vs.extract_patches(S_mce, [roi_obj])
        img = torch.tensor(patches[0], dtype=torch.float32) if (patches and patches[0] is not None) else torch.zeros((3, cfg.IMG_SIZE, cfg.IMG_SIZE))

        return img, f_vec, torch.tensor(label, dtype=torch.long)

def collate_eval(batch): return batch[0]