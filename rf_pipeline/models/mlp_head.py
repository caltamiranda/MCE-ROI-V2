# rf_pipeline/models/mlp_head.py

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    MLP head for the Engineered Feature Stream (Step 4A of the pipeline).

    Variant options:
        - "minimal" → ultra lightweight (edge deployment, lowest latency)
        - "medium"  → balanced capacity (recommended baseline)
        - "deep"    → higher capacity (for complex / low-SNR signals)

    Output:
        Raw logits → shape [batch, 2]   (binary: signal / no-signal)
        (Softmax is applied externally.)
    """

    def __init__(self, variant: str = "minimal"):
        super().__init__()
        self.variant = variant.lower()

        if self.variant == "minimal":
            self.net = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # logits
            )

        elif self.variant == "medium":
            self.net = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # logits
            )

        elif self.variant == "deep":
            self.net = nn.Sequential(
                nn.Linear(32, 256),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 2),  # logits
            )

        else:
            raise ValueError(
                f"Invalid MLP variant: {self.variant}. "
                "Use 'minimal', 'medium', or 'deep'."
            )

        print(f"[MLPHead] Loaded variant: {self.variant.upper()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Shape [batch, 32] — engineered 32-D feature vector.

        Returns:
            Tensor: Raw logits [batch, 2] (softmax to be applied externally).
        """
        return self.net(x)
