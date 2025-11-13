import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHead(nn.Module):
    """
    CNN head reforzada (BatchNorm + Dropout2d) para el stream visual.
    Input:  [B, C, 32, 32]  (C=1 o 3)
    Output: [B, 64] (embedding)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # [B,16,16,16]
            nn.Dropout2d(0.05),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # [B,32,8,8]
            nn.Dropout2d(0.05),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),     # [B,64,8,8]
        )

        self.embedding = nn.Sequential(
            nn.Flatten(start_dim=1),           # [B, 64*8*8]
            nn.Linear(64 * 8 * 8, embed_dim),  # [B, 64]
        )

        print(f"[CNNHead] in_channels={in_channels}, embed_dim={embed_dim} (BatchNorm+Dropout2d)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)
        return x
