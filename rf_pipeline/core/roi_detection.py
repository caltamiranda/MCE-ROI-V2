from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, label, find_objects, binary_dilation, binary_closing, generate_binary_structure

@dataclass
class ROI:
    y1: int; x1: int; y2: int; x2: int
    score: float
    snr_db: float = 0.0

class ROIDetector:
    """
    Detector Robusto basado en Estadística de Ruido (Z-Score / CFAR-like).
    No depende de umbrales absolutos, sino de desviaciones estándar sobre el fondo.
    """

    def __init__(
        self,
        # --- Parámetros Estadísticos ---
        method: str = "zscore",   # zscore (Recomendado) o adaptive (Legacy)
        z_thresh: float = 3.5,    # Umbral Z-Score: >3.0 detecta el 99.7% de anomalías
        background_window: int = 35, # Ventana grande para estimar el piso de ruido
        
        # --- Pre-procesamiento ---
        gauss_sigma: float = 1.0, # Suavizado leve para reducir picos de ruido
        
        # --- Post-procesamiento (Geometría) ---
        min_area: int = 10,       # Área mínima en píxeles
        min_ar: float = 0.1,      # Aspect Ratio mínimo
        max_ar: float = 20.0,     # Aspect Ratio máximo
        merge_dist: int = 3,      # Distancia para unir cajas fragmentadas
        
        # --- Legacy (Para compatibilidad si usas method='adaptive') ---
        adaptive_k: float = -0.2,
        adaptive_window: int = 15,
        margin_bins: int = 1,
        min_texture: float = 0.0
    ):
        self.method = method
        self.z_thresh = z_thresh
        self.bg_win = background_window
        self.gauss_sigma = gauss_sigma
        self.min_area = min_area
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.merge_dist = merge_dist
        
        # Legacy
        self.adaptive_k = adaptive_k
        self.adaptive_window = adaptive_window
        self.margin_bins = margin_bins

    def _compute_zscore_mask(self, img: np.ndarray) -> np.ndarray:
        """
        1. Estima el fondo usando filtro de mediana (ignora los picos de señal).
        2. Calcula la desviación estándar del ruido (MAD: Median Absolute Deviation).
        3. Genera mapa Z: (Img - Fondo) / Ruido.
        """
        # 1. Estimación robusta del fondo (Median Filter ignora señales cortas)
        # Usamos una ventana grande para capturar la tendencia del piso de ruido
        background = median_filter(img, size=self.bg_win)
        
        # 2. Estimación robusta de la desviación (ruido)
        # MAD = median(|x - median(x)|) -> sigma ~ 1.4826 * MAD
        diff = np.abs(img - background)
        noise_mad = median_filter(diff, size=self.bg_win)
        noise_sigma = 1.4826 * noise_mad
        
        # Evitar división por cero
        noise_sigma = np.maximum(noise_sigma, 1e-6)
        
        # 3. Mapa Z
        z_map = (img - background) / noise_sigma
        
        return z_map > self.z_thresh

    def detect(self, S_det: np.ndarray) -> List[ROI]:
        """
        Detecta ROIs en el espectrograma (o canal de gradiente).
        """
        H, W = S_det.shape
        
        # 1. Suavizado leve (Low-pass)
        img = gaussian_filter(S_det, sigma=self.gauss_sigma)
        
        # 2. Generar Máscara Binaria
        if self.method == "zscore":
            mask = self._compute_zscore_mask(img)
        else:
            # Fallback al método simple si se requiere
            from scipy.ndimage import uniform_filter
            mean = uniform_filter(img, size=self.adaptive_window)
            thr = mean + self.adaptive_k * (mean - img.min())
            mask = img > thr

# 3. Limpieza Morfológica (Unir fragmentos cercanos)
        
        # CAMBIO: Closing ANISOTRÓPICO (Solo Horizontal)
        # Usamos (1, 5) -> Une huecos de hasta 5 px en horizontal, 0 en vertical.
        mask = binary_closing(mask, structure=np.ones((1, 5))) 

        # Dilation ANISOTRÓPICA (Ya lo tenías, aseguramos que siga así)
        if self.merge_dist > 0:
            kernel_width = int(self.merge_dist * 2 + 1)
            structure = np.ones((1, kernel_width), dtype=int)
            mask = binary_dilation(mask, structure=structure)

        # 4. Etiquetado (Connected Components)
        labeled, num_feats = label(mask)
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
            
            # Filtros Geométricos
            if area < self.min_area: continue
            ar = w / (h + 1e-6)
            if not (self.min_ar <= ar <= self.max_ar): continue
            
            # Calcular score (intensidad media dentro de la caja original)
            score = float(np.mean(S_det[y1:y2, x1:x2]))
            
            # Ajuste fino de bordes (quitar el margen de dilatación si se quiere precisión)
            # Aquí lo dejamos tal cual para asegurar que cubra la señal completa
            
            rois.append(ROI(y1, x1, y2, x2, score))

        return sorted(rois, key=lambda r: r.score, reverse=True)