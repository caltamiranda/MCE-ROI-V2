import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

# Imports del pipeline
from core.visual_stream import VisualStream
from core.preprocessing import Preprocessor
from core.feature_engineering import FeatureEngineer
from core.roi_detection import AdaptiveROIDetector, ROI  # El nuevo detector SOTA
from models.hybrid_classifier import HybridRFClassifier

@torch.no_grad()
def dense_heatmap(
    iq: np.ndarray,
    model: HybridRFClassifier,
    fs: float = 10_000_000,
    nperseg: int = 32,
    noverlap: int = 16,
    tile: int = 32,
    stride_t: int = 8,  # Mantenido por compatibilidad de firma, aunque el detector es adaptativo
    stride_f: int = 8,
    prob_branch: str = "final",  # "final" usa consensus, "cnn" solo visual
    conf_threshold: float = 0.50 # Umbral para aceptar una detección como válida
):
    """
    Pipeline de Inferencia Avanzado:
    1. Preprocesamiento MCE (3 canales).
    2. Detección SOTA (Adaptive FCME + Hysteresis) -> Propone regiones candidatas.
    3. Extracción de Features (Estadísticas + Visuales).
    4. Clasificación Híbrida con Consenso (CNN + MLP + Atención).
    
    Devuelve:
      - heatmap: Representación visual de las probabilidades (opcional).
      - detections: Lista de tuplas (y1, x1, y2, x2, score).
    """
    # 1. Preparación de Herramientas
    pre = Preprocessor(fs=fs, nperseg=nperseg, noverlap=noverlap, mode="mce")
    vs = VisualStream(target_size=tile)
    fe = FeatureEngineer(fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Instanciamos el detector SOTA (Parámetros ajustados para alta sensibilidad)
    # high_sigma=4.0 asegura semillas fuertes, low_sigma=2.0 recupera bordes débiles
    detector = AdaptiveROIDetector(high_sigma=4.0, low_sigma=2.0, min_area=16, merge_dist=5)

    # 2. Computar Espectrograma
    # [3, H, W] -> Canales: Log-Mag, Gradiente, Energía Local
    S_mce = pre.compute(iq, show=False) 
    # Usamos el canal 1 (Gradiente) para la detección geométrica inicial
    S_det = S_mce[1] 
    
    C, H, W = S_mce.shape

    # 3. Paso 1: Propuesta de Regiones (SOTA ROI Detection)
    # En lugar de ventana deslizante ciega, buscamos qué parece señal.
    candidates: List[ROI] = detector.detect(S_det)

    if not candidates:
        # Retorno vacío seguro
        return np.zeros((H, W), np.float32), []

    # 4. Paso 2: Preparación de Batch para el Modelo Híbrido
    batch_imgs = []
    batch_feats = []
    valid_candidates = []

    # Convertimos IQ a tensor/numpy solo si es necesario para Features
    # FeatureEngineer espera IQ raw para sus cálculos estadísticos
    # Necesitamos mapear las ROIs espectrales al dominio del tiempo para extraer features
    # NOTA: FeatureEngineer.features_for_rois maneja esto internamente.
    
    # Extraemos características en bloque (más eficiente)
    roi_features_list = fe.features_for_rois(iq, S_det, candidates)
    # Extraemos parches visuales
    visual_patches = vs.extract_patches(S_mce, candidates)

    for i, roi in enumerate(candidates):
        # Chequeo de seguridad por si alguna extracción falló (bordes, tamaño 0)
        feat_obj = roi_features_list[i] if i < len(roi_features_list) else None
        patch = visual_patches[i] if i < len(visual_patches) else None

        if feat_obj is not None and patch is not None:
            batch_imgs.append(patch)
            batch_feats.append(feat_obj.features)
            valid_candidates.append(roi)

    if not batch_imgs:
        return np.zeros((H, W), np.float32), []

    # Crear Tensores [B, ...]
    X_vis = torch.from_numpy(np.stack(batch_imgs, 0))    # [B, 3, 32, 32]
    X_eng = torch.from_numpy(np.stack(batch_feats, 0))   # [B, 32]
    
    # Mover a dispositivo del modelo si es necesario
    device = next(model.parameters()).device
    X_vis = X_vis.to(device)
    X_eng = X_eng.to(device)

    # 5. Paso 3: Inferencia Híbrida
    if prob_branch == "final":
        # Usamos forward con sondas para obtener representaciones intermedias
        logits, probes = model.forward_with_probes(X_vis, X_eng)
        
        # Consenso: Fusiona la opinión de la red visual (pA), features (pB) y la fusión (pF)
        # Esto reduce drásticamente falsos positivos si una rama se confunde.
        probs = model.consensus_score(
            probes['pF'], probes['pA'], probes['pB'],
            wF=0.5, wA=0.25, wB=0.25
        )
    elif prob_branch == "cnn":
        logits = model(X_vis, X_eng)
        probs = F.softmax(logits, dim=1)[:, 1]
    else:
        raise ValueError("prob_branch debe ser 'final' o 'cnn'")

    probs = probs.cpu().numpy()

    # 6. Construcción de Resultados y Heatmap
    final_detections = []
    
    # Creamos un heatmap "sparse" (solo pintamos donde detectamos)
    # Esto es más informativo que una grilla de baja resolución.
    heatmap = np.zeros((H, W), dtype=np.float32)

    for i, roi in enumerate(valid_candidates):
        score = float(probs[i])
        
        # Filtrado final por confianza del modelo
        if score >= conf_threshold:
            # Guardamos detección: (y1, x1, y2, x2, score)
            final_detections.append((roi.y1, roi.x1, roi.y2, roi.x2, score))
            
            # Pintamos el heatmap con el score
            # (Opcional: podrías sumar si hay solapamiento, aquí reemplazamos max)
            heatmap[roi.y1:roi.y2, roi.x1:roi.x2] = np.maximum(
                heatmap[roi.y1:roi.y2, roi.x1:roi.x2], 
                score
            )

    # NMS (Non-Maximum Suppression)
    # Aunque el ROI detector ya une regiones, el modelo puede refinar scores.
    # Aplicamos un NMS ligero para limpiar solapamientos redundantes si los hubiera.
    if final_detections:
        boxes_np = np.array([d[:4] for d in final_detections], dtype=np.float32)
        scores_np = np.array([d[4] for d in final_detections], dtype=np.float32)
        
        keep_idxs = nms(boxes_np, scores_np, iou_thr=0.3)
        final_detections = [final_detections[i] for i in keep_idxs]

    return heatmap, final_detections

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr=0.3):
    """NMS estándar Vectorizado"""
    if len(boxes) == 0: return []
    
    x1 = boxes[:, 1]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    y2 = boxes[:, 2]
    area = (x2 - x1) * (y2 - y1)
    
    idxs = np.argsort(scores)[::-1] # Orden descendente
    keep = []
    
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        union = area[i] + area[idxs[1:]] - inter
        iou = inter / (union + 1e-6)
        
        # Quedarse solo con los que tienen IoU bajo con el actual
        idxs = idxs[1:][iou <= iou_thr]
        
    return keep