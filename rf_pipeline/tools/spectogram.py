import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import time as time

# === CONFIG IGUAL AL PIPELINE ===
FS = 32000 # debe coincidir con tu señal
NPERSEG = 32
NOVERLAP = 16

# === CARGAR IQ (ejemplo .npy) ===
iq = np.load(r"synthetic_signal_0.npy")
# iq debe ser COMPLEX1D: ej. array([0.013+0.02j, ...])
a= time.time()
# === GENERAR STFT ===
f, t, Zxx = stft(iq, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
S = np.abs(Zxx).astype(np.float32)
S = np.log1p(S)   # ← EXACTAMENTE COMO EN TU PREPROCESSOR (log1p)
b=time.time()
print(b-a)
# === VISUALIZAR ===
plt.figure(figsize=(10,4))
plt.imshow(S, aspect="auto", origin="lower", cmap="viridis")
plt.title("Espectrograma log|STFT| (como C0 del pipeline MCE)")
plt.colorbar(label="log-magnitud")
plt.xlabel("Tiempo (bins)")
plt.ylabel("Frecuencia (bins)")
plt.show()
