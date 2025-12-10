import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Cargar imagen (Escala de grises)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Convertir a Tensor [1, 64, 64]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Cargar Máscara (Target) desde Labels YOLO
        mask_target = np.zeros((64, 64), dtype=np.float32)
        txt_name = img_name.replace('.png', '.txt')
        lbl_path = os.path.join(self.lbl_dir, txt_name)
        
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    # YOLO format: cls, cx, cy, w, h (normalized)
                    _, cx, cy, w, h = map(float, parts[:5])
                    
                    x1 = int((cx - w/2) * 64)
                    y1 = int((cy - h/2) * 64)
                    x2 = int((cx + w/2) * 64)
                    y2 = int((cy + h/2) * 64)
                    
                    # Clampear a bordes
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(64, x2), min(64, y2)
                    
                    # Dibujar caja en la máscara target (valor 1.0)
                    mask_target[y1:y2, x1:x2] = 1.0
                    boxes.append([x1, y1, x2, y2])

        mask_tensor = torch.from_numpy(mask_target).unsqueeze(0)
        
        return img_tensor, mask_tensor, boxes, img_name