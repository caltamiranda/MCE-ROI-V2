# config.py
import os

# --- Rutas ---
ROOT_DIR = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\raw_iq_hdf5"
TRAIN_H5 = os.path.join(ROOT_DIR, "train", "data.h5")
VAL_H5   = os.path.join(ROOT_DIR, "val", "data.h5")
TEST_H5  = os.path.join(ROOT_DIR, "test", "data.h5")

# --- Parámetros de Señal (Deben coincidir con los usados al crear el H5) ---
FS = 10_000_000      # Frecuencia de muestreo (ajustar según tu dataset TorchSig)
CENTER_FREQ = 0      # Usualmente 0 para banda base compleja

# --- Parámetros STFT / Preprocesamiento ---
NPERSEG = 32
NOVERLAP = 16
NFFT = 32            # Generalmente igual a nperseg
IMG_SIZE = 32        # Tamaño del parche para la CNN

# --- Entrenamiento ---
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cpu"