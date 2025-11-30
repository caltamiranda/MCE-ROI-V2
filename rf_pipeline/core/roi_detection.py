from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.ndimage import label, find_objects, binary_dilation, generate_binary_structure
from scipy.ndimage import morphology

@dataclass
class ROI:
    y1: int; x1: int; y2: int; x2: int
    score: float
    snr_db: float = 0.0

class AdaptiveROIDetector:
    """
    SOTA Statistical Detector: FCME (Forward Consecutive Mean Excision) + Hysteresis.
    
    1. Auto-ajuste: Estima iterativamente el piso de ruido real excluyendo señales.
    2. Precisión: Usa umbralización dual (Histéresis) para expandir detecciones fuertes 
       hacia sus bordes débiles sin capturar ruido aislado.
    """

    def __init__(
        self,
        # --- Parámetros de Histéresis (Auto-ajuste) ---
        high_sigma: float = 4.0,  # Semillas fuertes (Certeza absoluta)
        low_sigma: float = 2.0,   # Bordes débiles (Alta sensibilidad)
        
        # --- Parámetros FCME ---
        max_iterations: int = 10,
        convergence_tol: float = 0.05,
        
        # --- Geometría ---
        min_area: int = 10,
        merge_dist: int = 5,      # Distancia de fusión
    ):
        self.high_sigma = high_sigma
        self.low_sigma = low_sigma
        self.max_iters = max_iterations
        self.tol = convergence_tol
        self.min_area = min_area
        self.merge_dist = merge_dist

    def _fcme_noise_estimation(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Forward Consecutive Mean Excision (Simplificado y Vectorizado).
        Encuentra mu y sigma del RUIDO PURO iterativamente.
        """
        # Aplanar y ordenar intensidades
        arr = img.ravel()
        # Submuestreo para velocidad si es muy grande
        if arr.size > 100000:
            arr = np.random.choice(arr, 100000, replace=False)
        
        # Inicialización robusta usando mediana (menos sensible a outliers que mean)
        mu = np.median(arr)
        sigma = 1.4826 * np.median(np.abs(arr - mu)) # MAD sigma
        
        # Iteración (Auto-ajuste)
        for _ in range(self.max_iters):
            # Criterio de corte actual: mu + 3*sigma (asumiendo que más allá es señal)
            cutoff = mu + 3.0 * sigma
            
            # Seleccionar píxeles que parecen ruido
            noise_pixels = arr[arr < cutoff]
            
            if noise_pixels.size == 0: break # Evitar error si todo es señal
            
            new_mu = np.mean(noise_pixels)
            new_sigma = np.std(noise_pixels)
            
            # Chequear convergencia
            if abs(new_sigma - sigma) / (sigma + 1e-9) < self.tol:
                mu, sigma = new_mu, new_sigma
                break
            
            mu, sigma = new_mu, new_sigma
            
        return mu, sigma

    def detect(self, S_det: np.ndarray) -> List[ROI]:
        """
        Detecta ROIs usando crecimiento de regiones por histéresis.
        """
        H, W = S_det.shape
        
        # 1. Estimación Inteligente del Piso de Ruido (FCME)
        noise_mu, noise_sigma = self._fcme_noise_estimation(S_det)
        noise_sigma = max(noise_sigma, 1e-9) # Evitar div/0
        
        # Normalizar imagen a espacio Z-score real basado en ruido limpio
        z_img = (S_det - noise_mu) / noise_sigma
        
        # 2. Umbralización Dual (Histéresis)
        # a) Semillas Fuertes: Señales obvias
        strong_mask = z_img > self.high_sigma 
        
        # b) Máscara Débil: Todo lo que podría ser señal
        weak_mask = z_img > self.low_sigma
        
        # 3. Reconstrucción Geodésica (Morphological Reconstruction)
        # Expandir las semillas 'strong' dentro del territorio 'weak'.
        # Solo sobrevivirá el 'weak' que toque a un 'strong'.
        # Esto elimina ruido débil aislado pero conserva colas de señales fuertes.
        
        final_mask = morphology.binary_dilation(
            strong_mask, 
            mask=weak_mask, 
            iterations=-1, # Iterar hasta que no cambie (reconstrucción completa)
            structure=generate_binary_structure(2, 2) # Conectividad-8
        )
        
        # 4. Post-Procesamiento (Unir fragmentos cercanos)
        if self.merge_dist > 0:
            # Closing horizontal agresivo (asumiendo señales continuas en tiempo)
            final_mask = morphology.binary_closing(final_mask, structure=np.ones((1, 9)))
            # Dilatación general para margen de seguridad
            k = int(self.merge_dist * 2 + 1)
            final_mask = binary_dilation(final_mask, structure=np.ones((k, k)))

        # 5. Extracción de Objetos
        labeled, num_feats = label(final_mask)
        if num_feats == 0:
            return []

        rois = []
        objects = find_objects(labeled)
        
        for idx, slc in enumerate(objects):
            if slc is None: continue
            y1, y2 = slc[0].start, slc[0].stop
            x1, x2 = slc[1].start, slc[1].stop
            
            h, w = y2 - y1, x2 - x1
            area = h * w
            
            if area < self.min_area: continue
            
            # Score: Promedio Z-score de la región (indica qué tan fuerte es respecto al ruido)
            roi_z = np.mean(z_img[y1:y2, x1:x2])
            
            # Estimación rápida de SNR db (basada en el piso estimado)
            # Potencia Señal ~ roi_mean^2, Potencia Ruido ~ noise_sigma^2 (aprox en dominio lineal)
            # Como S_det ya suele ser log o gradiente, usamos roi_z como proxy de confianza.
            
            rois.append(ROI(y1, x1, y2, x2, score=float(roi_z)))

        # Ordenar por confianza
        return sorted(rois, key=lambda r: r.score, reverse=True)