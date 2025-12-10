import torch
import torch.nn as nn

class Spectrogram1DCNN(nn.Module):
    def __init__(self):
        super(Spectrogram1DCNN, self).__init__()
        
        # --- RAMA TEMPORAL (Eje X: Tiempo) ---
        # Entrada: 64 canales de frecuencia (las filas de la imagen)
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), # Salida: Probabilidad por columna de tiempo
            nn.Sigmoid()
        )

        # --- RAMA FRECUENCIA (Eje Y: Frecuencia) ---
        # Entrada: 64 canales de tiempo (las columnas de la imagen)
        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), # Salida: Probabilidad por fila de frecuencia
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, 1, 64, 64] (Grayscale)
        x = x.squeeze(1) # [Batch, 64, 64]
        
        # 1. Procesar Tiempo: Tratamos las 64 filas de freq como canales
        # Input: [Batch, 64 (Canales), 64 (Largo)]
        time_mask = self.time_conv(x) # -> [Batch, 1, 64]
        
        # 2. Procesar Frecuencia: Transponemos para que el Tiempo sean los canales
        x_trans = x.permute(0, 2, 1) # [Batch, 64, 64]
        freq_mask = self.freq_conv(x_trans) # -> [Batch, 1, 64]
        
        # 3. Combinar para crear máscara 2D (Producto externo)
        # Multiplicamos (Batch, 1, 64) x (Batch, 64, 1) -> (Batch, 64, 64)
        output_map = torch.matmul(freq_mask.permute(0, 2, 1), time_mask)
        
        return output_map, time_mask, freq_mask