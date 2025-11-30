import torch
import torch.nn as nn
import torch.nn.functional as F

class SNRAwareFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = 'mean'):
        """
        gamma: Enfoca el entrenamiento en ejemplos difíciles (predicciones con baja probabilidad).
               Valor estándar: 2.0.
        alpha: Factor de balance para la clase positiva (ej: señal=0.75, ruido=0.25).
        reduction: 'mean' (promedio) o 'sum' (suma total).
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, snr_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: [B, 2] Raw logits (sin softmax previa).
            targets: [B] Etiquetas enteras {0, 1}.
            snr_weights: [B] (Opcional) Tensor de floats. 
                         Factor multiplicativo basado en SNR. 
                         Ej: 1.0 para ruido/señal débil, >1.0 para señal fuerte.
                         Esto fuerza al modelo a aprender características obvias primero.
        """
        # Calcular probabilidades
        probs = F.softmax(logits, dim=1)               # [B, 2]
        
        # Seleccionar la probabilidad de la clase correcta (pt)
        # targets.view(-1, 1) convierte [B] -> [B, 1] para gather
        pt = probs.gather(1, targets.view(-1, 1)).view(-1) # [B]

        # Calcular pesos Alpha (balance de clases)
        # Si target=1 usa self.alpha, si target=0 usa (1 - self.alpha)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # Calcular término Focal (modulador de dificultad)
        focal_term = (1 - pt) ** self.gamma

        # Pérdida base (Cross Entropy ponderada)
        # Se añade 1e-8 para estabilidad numérica en log
        loss = -alpha_t * focal_term * torch.log(pt + 1e-8)

        # --- Inyección de SNR ---
        if snr_weights is not None:
            # Aseguramos que los pesos tengan la misma forma que la loss [B]
            if snr_weights.shape != loss.shape:
                snr_weights = snr_weights.view_as(loss)
            loss = loss * snr_weights

        # Reducción final
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss