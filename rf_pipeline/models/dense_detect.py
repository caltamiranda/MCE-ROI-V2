import numpy as np, torch
import torch.nn.functional as F
from typing import List, Tuple
from visual_stream import VisualStream
from preprocessing import Preprocessor
from models.hybrid_classifier import HybridRFClassifier
# ↑ Usa tu visual stream (normaliza parches) y preprocesador MCE; el modelo híbrido ya lo usas en train.  :contentReference[oaicite:0]{index=0}

@torch.no_grad()
def dense_heatmap(
    iq: np.ndarray,
    model: HybridRFClassifier,
    fs: float = 1_000_000,
    nperseg: int = 32,
    noverlap: int = 16,
    tile: int = 32,
    stride_t: int = 8,
    stride_f: int = 8,
    prob_branch: str = "final"  # "final" usa consensus, o "cnn" para rama visual
):
    """
    Devuelve:
      - heatmap (H', W') con prob de 'señal'
      - listas (y1,x1,y2,x2,score) después de NMS
    """
    pre = Preprocessor(fs=fs, nperseg=nperseg, noverlap=noverlap, mode="mce")
    S = pre.compute(iq, show=False)   # [3,H,W]
    C, H, W = S.shape
    vs = VisualStream(target_size=tile)  # normalización visual idéntica a tu pipeline  :contentReference[oaicite:1]{index=1}

    # recopilar parches [N,3,32,32]
    coords = []
    patches = []
    for y1 in range(0, H - tile + 1, stride_f):
        for x1 in range(0, W - tile + 1, stride_t):
            patch = S[:, y1:y1+tile, x1:x1+tile]
            if patch.size == 0: continue
            # ya está 3×tile×tile → normalizado por canal (log1p en pre/VisualStream)
            patches.append(patch.astype(np.float32))
            coords.append((y1, x1, y1+tile, x1+tile))

    if not patches:
        return np.zeros((H//stride_f+1, W//stride_t+1), np.float32), []

    X_vis = torch.from_numpy(np.stack(patches, 0))               # [N,3,32,32]
    X_eng = torch.zeros(len(patches), 32, dtype=torch.float32)   # sin features de ingeniería aquí

    # puntuación
    if prob_branch == "final":
        # Usa consenso ya disponible en tu modelo (forward_with_probes)
        logits_f, pA, pB, pF = model.forward_with_probes(X_eng, X_vis)
        probs = model.consensus_score(pF, pA, pB, wF=0.5, wA=0.25, wB=0.25,
                                      agree_delta=0.35, min_branch=0.50)
    elif prob_branch == "cnn":
        logits = model(X_eng, X_vis)           # usa forward normal
        probs = F.softmax(logits, dim=1)[:,1]  # prob clase 'señal'
    else:
        raise ValueError("prob_branch debe ser 'final' o 'cnn'")

    probs = probs.cpu().numpy().astype(np.float32)

    # __heatmap__ discretizando a la rejilla de strides
    Hh = (H - tile)//stride_f + 1
    Wh = (W - tile)//stride_t + 1
    heat = np.zeros((Hh, Wh), np.float32)
    k = 0
    for iy in range(Hh):
        for ix in range(Wh):
            if k < len(probs):
                heat[iy, ix] = probs[k]
                k += 1

    # NMS sencilla (IoU)
    boxes = np.array(coords, dtype=np.int32)
    scores = probs
    keep = nms(boxes, scores, iou_thr=0.3, score_thr=0.5)
    kept = [(int(*boxes[i][0:1]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]), float(scores[i])) for i in keep]
    # Nota: arriba puedes adaptar formato; aquí devuelvo (y1,x1,y2,x2,score)

    return heat, kept

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr=0.3, score_thr=0.5):
    idxs = np.where(scores >= score_thr)[0]
    idxs = idxs[np.argsort(scores[idxs])[::-1]]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
        iou = iou_with(i, idxs[1:], boxes)
        idxs = idxs[1:][iou <= iou_thr]
    return keep

def iou_with(i, js, boxes):
    y1,x1,y2,x2 = boxes[i]
    area_i = (y2-y1)*(x2-x1)
    out = np.zeros(len(js), np.float32)
    for k,j in enumerate(js):
        yy1,xx1,yy2,xx2 = boxes[j]
        inter_y1 = max(y1, yy1); inter_x1 = max(x1, xx1)
        inter_y2 = min(y2, yy2); inter_x2 = min(x2, xx2)
        inter = max(0, inter_y2-inter_y1) * max(0, inter_x2-inter_x1)
        area_j = (yy2-yy1)*(xx2-xx1)
        union = area_i + area_j - inter + 1e-6
        out[k] = inter/union
    return out
