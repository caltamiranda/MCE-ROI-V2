# train_h5.py — Entrenamiento desde TorchSigDataset/data.h5
import os
import random
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split
from models.hybrid_classifier import HybridRFClassifier
from losses import FocalLoss
from scipy.signal import stft

# =========================
# Configuración
# =========================
DATASET_PATH = r"C:\Users\User\Documents\GitHub\torchsig\scripts\TorchSigDataset\train\data.h5"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.7
VAL_SPLIT = 0.1
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Dataset
# =========================
class H5ROIDataset(Dataset):
    """
    Carga ejemplos (I/Q) desde data.h5.
    Genera: (features[32], patch[3x32x32], label)
    """
    def __init__(self, h5_path: str, use_fraction: float = 1.0):
        super().__init__()
        self.file = h5py.File(h5_path, "r")
        keys = [k for k in self.file.keys() if k.isdigit()]
        limit = int(len(keys) * use_fraction)
        self.keys = keys[:limit]
        print(f"[INFO] Dataset cargado con {len(self.keys)} muestras ({use_fraction*100:.0f}%)")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        group = self.file[key]
        data = np.array(group["data"])

        # --- Reconstrucción IQ ---
        if np.iscomplexobj(data):
            iq = data
        elif data.ndim == 1:
            iq = data[::2] + 1j * data[1::2]
        elif data.ndim == 2 and data.shape[1] == 2:
            iq = data[:, 0] + 1j * data[:, 1]
        else:
            raise ValueError(f"Formato desconocido: {data.shape}")

        amp = np.abs(iq).astype(np.float32)

        # --- Feature Engineering (32D) ---
        feats = np.array([
            np.mean(amp), np.std(amp), np.max(amp), np.min(amp),
            np.median(amp), np.var(amp), np.mean(np.diff(amp)**2),
            np.percentile(amp, 25), np.percentile(amp, 75),
            np.mean(np.abs(amp - np.mean(amp)))
        ], dtype=np.float32)
        feats = np.pad(feats, (0, 32 - len(feats)))  # llenar hasta 32D
        feats = torch.tensor(feats, dtype=torch.float32)

        # --- Patch visual (espectrograma) ---
        _, _, Zxx = stft(iq, fs=10e6, nperseg=128, noverlap=64)
        Sxx = np.abs(Zxx)
        Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-8)
        Sxx = np.expand_dims(Sxx, axis=0)  # [1, H, W]
        Sxx = torch.tensor(Sxx, dtype=torch.float32)
        Sxx = torch.nn.functional.interpolate(
            Sxx.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze(0)
        Sxx = Sxx.repeat(3, 1, 1)  # [3, 32, 32]

        # --- Etiqueta ---
        try:
            meta = group["metadata/0"]
            class_name = meta["class_name"][()].decode("utf-8").lower()
        except Exception:
            class_name = "unknown"
        y = 1 if any(x in class_name for x in ["qam", "psk", "ask"]) else 0
        y = torch.tensor(y, dtype=torch.long)

        return feats, Sxx, y


# =========================
# Entrenamiento
# =========================
def train_loop():
    ds_full = H5ROIDataset(DATASET_PATH, use_fraction=TRAIN_FRACTION)
    val_len = max(1, int(len(ds_full) * VAL_SPLIT))
    train_len = len(ds_full) - val_len
    ds_train, ds_val = random_split(
        ds_full, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modelo DualStream (Hybrid) ---
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

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    best_val_loss, best_val_acc = float("inf"), 0.0
    epochs_no_improve = 0

    print(f"[INFO] Train samples: {train_len} | Val samples: {val_len}")
    for epoch in range(1, EPOCHS + 1):
        # --- Entrenamiento ---
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x_eng, x_vis, y in dl_train:
            x_eng, x_vis, y = x_eng.to(DEVICE), x_vis.to(DEVICE), y.to(DEVICE)
            logits = model(x_vis, x_eng)     # <-- orden correcto
            loss = criterion(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        # --- Validación ---
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for x_eng, x_vis, y in dl_val:
                x_eng, x_vis, y = x_eng.to(DEVICE), x_vis.to(DEVICE), y.to(DEVICE)
                logits = model(x_vis, x_eng)
                loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += y.size(0)

        va_loss /= va_total
        va_acc = va_correct / va_total

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"loss={tr_loss:.4f} | acc={tr_acc*100:.2f}% | "
              f"val_loss={va_loss:.4f} | val_acc={va_acc*100:.2f}%")

        # --- Early stopping ---
        if va_loss < best_val_loss - 1e-5:
            best_val_loss, best_val_acc = va_loss, va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[EarlyStopping] No mejora en {PATIENCE} épocas.")
                break

    # --- Guardar mejor modelo ---
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "hybrid_h5_best.pt")
    torch.save(best_state, ckpt_path)
    print(f"[OK] Modelo guardado en: {ckpt_path}")
    print(f"[SUMMARY] Mejor Val Acc: {best_val_acc*100:.2f}% | Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_loop()
