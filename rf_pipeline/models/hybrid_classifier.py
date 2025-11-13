# rf_pipeline/models/dual_stream_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Pequeños bloques reutilizables
# ------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.do   = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(64, 64), dropout=0.1, bn=True):
        super().__init__()
        layers = []
        d_prev = in_dim
        for d in hidden_dims:
            layers += [
                nn.Linear(d_prev, d, bias=not bn),
                nn.BatchNorm1d(d) if bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            d_prev = d
        self.net = nn.Sequential(*layers)
        self.out_dim = d_prev

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# DualStreamNet
# ------------------------------------------------------------
class HybridRFClassifier(nn.Module):
    """
    Entradas:
      - x_img: tensor [N, 1, H, W]  (parches ROI normalizados a [0,1] o z-score)
      - x_feat: tensor [N, F]       (por defecto F=32 engineered features)
    Salida:
      - logits: [N, C]  (C = n_clases)
    """
    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 32,
        img_in_ch: int = 1,
        cnn_channels=(16, 32, 64),
        cnn_dropout=0.05,
        mlp_hidden=(64, 64),
        mlp_dropout=0.10,
        fusion_hidden=(128, ),
        fusion_dropout=0.10,
        bn=True,
    ):
        super().__init__()

        # ----- Rama visual (CNN + GAP) -----
        c = [img_in_ch] + list(cnn_channels)
        blocks = []
        for i in range(1, len(c)):
            blocks.append(ConvBNReLU(c[i-1], c[i], k=3, s=1, p=1, dropout=cnn_dropout))
            # downsample suave cada 2 bloques o si lo prefieres siempre:
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.cnn = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.cnn_out = cnn_channels[-1]

        # ----- Rama de features (MLP) -----
        self.mlp = MLP(in_dim=feat_dim, hidden_dims=mlp_hidden, dropout=mlp_dropout, bn=bn)
        self.mlp_out = self.mlp.out_dim

        # ----- Fusión -----
        fusion_in = self.cnn_out + self.mlp_out
        f_layers = []
        d_prev = fusion_in
        for d in fusion_hidden:
            f_layers += [
                nn.Linear(d_prev, d, bias=not bn),
                nn.BatchNorm1d(d) if bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity(),
            ]
            d_prev = d
        self.fusion = nn.Sequential(*f_layers)
        self.head = nn.Linear(d_prev, num_classes)

        # Inicialización razonable
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- Proyección CNN con GAP ----
    def _visual_proj(self, x_img):
        h = self.cnn(x_img)              # [N, C, h, w]
        h = self.gap(h).squeeze(-1).squeeze(-1)  # [N, C]
        return h

    def forward(self, x_img, x_feat):
        """
        Devuelve logits. Usa CrossEntropyLoss externamente.
        """
        # seguridad de shapes
        assert x_img.dim() == 4, "x_img debe ser [N, 1, H, W]"
        assert x_feat.dim() == 2, "x_feat debe ser [N, F]"

        pA = self._visual_proj(x_img)   # rama A (visual)
        pB = self.mlp(x_feat)           # rama B (features)
        pF_in = torch.cat([pA, pB], dim=1)
        pF = self.fusion(pF_in) if len(self.fusion) > 0 else pF_in
        logits = self.head(pF)
        return logits

    @torch.no_grad()
    def forward_with_probes(self, x_img, x_feat):
        """
        Devuelve (logits, probes) donde probes = dict(pA, pB, pF)
        para análisis de interpretabilidad.
        """
        pA = self._visual_proj(x_img)
        pB = self.mlp(x_feat)
        pF_in = torch.cat([pA, pB], dim=1)
        pF = self.fusion(pF_in) if len(self.fusion) > 0 else pF_in
        logits = self.head(pF)
        return logits, {"pA": pA, "pB": pB, "pF": pF}
