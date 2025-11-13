import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import stft

# === MISMOS PARÁMETROS DEL PIPELINE ===
FS = 1_000_000     # debe coincidir con el de make_rf_dataset.py
NPERSEG = 32
NOVERLAP = 16

# Cambia por cualquier sample de rf_dataset
FILE = r"rf_dataset\test\ofdm\sample_000011.npy"

# --- Cargar IQ como COMPLEX ---
iq2 = np.load(FILE)  # [N,2] float32
iq = iq2[:,0] + 1j*iq2[:,1]  # → complejo 1D

# --- STFT EXACTAMENTE IGUAL AL PIPELINE ---A
f, t, Zxx = stft(iq, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
S = np.abs(Zxx).astype(np.float32)
S = np.log1p(S)  # mismo log1p que usa tu preprocesador

print("S shape:", S.shape)  # → (freq_bins, time_bins)

# --- Visualizar ---
plt.figure(figsize=(10,4))
plt.imshow(S, aspect="auto", origin="lower", cmap="viridis")

plt.title(f"Espectrograma log|STFT| — {os.path.basename(os.path.dirname(FILE)).upper()}")

plt.xlabel("Tiempo (bins)")
plt.ylabel("Frecuencia (bins)")
plt.colorbar(label="Magnitud log1p")
plt.show()
