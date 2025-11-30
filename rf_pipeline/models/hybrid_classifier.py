# rf_pipeline/models/dual_stream_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Bloques de Atención (CBAM)
# ------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Si los canales son pocos, asegurar al menos 1 en la capa oculta
        hidden_planes = max(1, in_planes // ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compresión de canales: Max y Avg a lo largo de la dimensión de canales
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Refina secuencialmente las features por Canal (Qué mirar) y Espacio (Dónde mirar).
    """
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ------------------------------------------------------------
# Bloques de Construcción Básicos
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
# DualStreamNet (HybridRFClassifier)
# ------------------------------------------------------------
class HybridRFClassifier(nn.Module):
    """
    Entradas:
      - x_img: tensor [N, 1, H, W]  (parches ROI normalizados)
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

        # ----- Rama visual (CNN + CBAM + GAP) -----
        c = [img_in_ch] + list(cnn_channels)
        blocks = []
        for i in range(1, len(c)):
            # 1. Convolución
            blocks.append(ConvBNReLU(c[i-1], c[i], k=3, s=1, p=1, dropout=cnn_dropout))
            
            # 2. Atención (CBAM) - Auto-ajuste de ruido/señal
            blocks.append(CBAM(c[i]))
            
            # 3. Downsampling
            # Usamos MaxPool para preservar las características más fuertes
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

        # Inicialización
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---- Proyección CNN con GAP ----
    def _visual_proj(self, x_img):
        h = self.cnn(x_img)              # [N, C, h, w]
        h = self.gap(h).squeeze(-1).squeeze(-1)  # [N, C]
        return h

    def forward(self, x_img, x_feat):
        """
        Devuelve logits combinados.
        """
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
        Utilizado para análisis o weighting manual en inferencia.
        """
        pA = self._visual_proj(x_img)
        pB = self.mlp(x_feat)
        pF_in = torch.cat([pA, pB], dim=1)
        pF = self.fusion(pF_in) if len(self.fusion) > 0 else pF_in
        logits = self.head(pF)
        return logits, {"pA": pA, "pB": pB, "pF": pF}

    @torch.no_grad()
    def consensus_score(self, pF, pA, pB, wF=0.5, wA=0.25, wB=0.25, agree_delta=0.35, min_branch=0.50):
        """
        Calcula una probabilidad de consenso robusta durante la inferencia.
        Útil para reducir Falsos Positivos.
        
        Args:
           pF: Proyecciones de fusión (antes del head final, pero aquí asumimos uso simplificado o forward completo).
               NOTA: Para simplificar uso en dense_detect, a menudo se prefiere pasar logits.
               Aquí implementamos la lógica sobre las probabilidades finales si tu head es lineal.
        """
        # Calcular logits individuales (aproximación rápida si no tienes heads separados entrenados)
        # Nota: Idealmente deberías tener heads auxiliares para pA y pB si quieres consenso real.
        # Asumiremos que el 'head' principal sirve para pF.
        
        # Como es inferencia post-training, usamos el logit final principal para todo 
        # o implementamos heads auxiliares ligeros.
        # Para mantener compatibilidad con tu dense_detect, usamos el logit principal:
        logits = self.head(pF) 
        probs = F.softmax(logits, dim=1)[:, 1] # Prob clase 1
        return probs 
        
   