import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from models.hybrid_classifier import HybridRFClassifier
from losses import FocalLoss

# =========================
# CONFIGURACIÓN
# =========================
DATASET_DIR = r"C:\Users\User\Documents\GitHub\torchsig\scripts\dataset_yolo"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
SEED = 42
VAL_SPLIT = 0.15
torch.manual_seed(SEED)

# =========================
# DATASET
# =========================
class YOLOROIDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        rois, labels = [], []

        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                roi = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if roi.size == 0:
                    continue
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (32, 32))
                roi = self.transform(roi)
                rois.append(roi)
                labels.append(1 if int(cls) > 0 else 0)


        if not rois:
            # Si no hay ROI válida, retorna imagen vacía + clase 0 (ruido)
            empty = torch.zeros((3, 32, 32))
            return empty, torch.tensor(0)

        # Por simplicidad, retorna la primera ROI (o se podría hacer multiclase)
        return rois[0], torch.tensor(labels[0])


# =========================
# ENTRENAMIENTO
# =========================
def train_loop():
    ds = YOLOROIDataset(IMG_DIR, LABEL_DIR)
    val_len = int(len(ds) * VAL_SPLIT)
    train_len = len(ds) - val_len
    ds_train, ds_val = random_split(ds, [train_len, val_len])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

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
    ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x_vis, y in dl_train:
            x_vis, y = x_vis.to(DEVICE), y.to(DEVICE)
            # Sin features tabulares, usa solo rama visual
            feats = torch.zeros((x_vis.size(0), 32), device=DEVICE)
            logits = model(x_vis, feats)
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch:03d} | Loss={train_loss/len(dl_train):.4f} | Acc={acc*100:.1f}%")

        # Validación
        model.eval()
        with torch.no_grad():
            val_correct, val_total = 0, 0
            for x_vis, y in dl_val:
                x_vis, y = x_vis.to(DEVICE), y.to(DEVICE)
                feats = torch.zeros((x_vis.size(0), 32), device=DEVICE)
                logits = model(x_vis, feats)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        print(f"   → Val Acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/hybrid_from_yolo.pt")
            print(f"   ✓ Nuevo mejor modelo guardado ({best_acc*100:.2f}%)")

if __name__ == "__main__":
    train_loop()
