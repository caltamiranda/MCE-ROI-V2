import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = 'mean'):
        """
        gamma: enfoca en ejemplos difíciles (2.0 es un estándar sólido)
        alpha: peso para la clase positiva (ej: señal = 0.75, no-señal = 0.25)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B,2]  (sin softmax)
        targets: [B] con valores enteros {0 o 1}
        """
        probs = F.softmax(logits, dim=1)               # [B,2]
        pt = probs[range(len(targets)), targets]       # prob. predicha correcta

        focal_weight = (self.alpha * (1 - pt) ** self.gamma)
        loss = -focal_weight * torch.log(pt + 1e-8)

        return loss.mean() if self.reduction == 'mean' else loss.sum()
