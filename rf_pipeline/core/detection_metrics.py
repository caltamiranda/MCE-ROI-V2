# detection_metrics.py
import numpy as np
import torch
from typing import List, Dict, Tuple

def calculate_iou(box1, box2):
    """
    Calcula Intersection over Union (IoU) entre dos cajas.
    Formato caja: [x1, y1, x2, y2]
    """
    # Coordenadas de la intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-12)

def compute_map(
    pred_boxes: List[np.ndarray], 
    pred_scores: List[np.ndarray], 
    gt_boxes: List[np.ndarray], 
    iou_threshold: float = 0.5
) -> float:
    """
    Calcula el mAP@IoU (Average Precision) para una sola clase (Señal).
    
    Args:
        pred_boxes: Lista de [N, 4] arrays (x1, y1, x2, y2) por cada imagen.
        pred_scores: Lista de [N] arrays (confianza 0-1) por cada imagen.
        gt_boxes: Lista de [M, 4] arrays (Ground Truth) por cada imagen.
        iou_threshold: Umbral para considerar una detección correcta (TP).
    """
    true_positives = []
    scores = []
    num_gt_total = 0

    # Procesar imagen por imagen
    for i in range(len(gt_boxes)):
        boxes = pred_boxes[i]
        conf = pred_scores[i]
        gts = gt_boxes[i]
        num_gt_total += len(gts)

        if len(boxes) == 0:
            continue

        # Ordenar predicciones por confianza descendente
        idx_sorted = np.argsort(-conf)
        boxes = boxes[idx_sorted]
        conf = conf[idx_sorted]
        
        matched_gt = np.zeros(len(gts), dtype=bool)

        for b_idx, box in enumerate(boxes):
            scores.append(conf[b_idx])
            
            if len(gts) == 0:
                true_positives.append(0)
                continue

            # Calcular IoU con todos los GT de esta imagen
            ious = [calculate_iou(box, gt) for gt in gts]
            best_iou = max(ious)
            best_gt_idx = np.argmax(ious)

            if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
                true_positives.append(1)
                matched_gt[best_gt_idx] = True
            else:
                true_positives.append(0) # FP

    if num_gt_total == 0:
        return 0.0

    # Calcular Precision-Recall Curve
    scores = np.array(scores)
    true_positives = np.array(true_positives)
    
    # Ordenar globalmente por score
    idx = np.argsort(-scores)
    true_positives = true_positives[idx]
    
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(1 - true_positives)
    
    recalls = tp_cumsum / num_gt_total
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
    
    # Calcular AP (Area bajo la curva P-R usando método de interpolación de 11 puntos o integración)
    # Aquí usamos integración simple:
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Suavizar precisión (envelope)
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return float(ap)