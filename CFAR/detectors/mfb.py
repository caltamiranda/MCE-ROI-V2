import numpy as np
from skimage.feature import match_template
from skimage.morphology import local_maxima
from skimage.measure import label, regionprops

class MatchedFilterBankDetector:
    def __init__(self, threshold=0.6):
        """
        Detector basado en Banco de Filtros (Template Matching 2D).
        Crea plantillas sintéticas de formas de señal comunes.
        
        Args:
            threshold (float): Umbral de correlación (0 a 1). 
                               Si la imagen se parece más de X% a la plantilla, es detección.
        """
        self.threshold = threshold
        self.templates = self._generate_template_bank()

    def _generate_template_bank(self):
        """
        Genera 'moldes' de señales típicas en espectrogramas de 64x64.
        """
        templates = []
        
        # Definir formas típicas (Alto, Ancho)
        shapes = [
            (10, 4),   # Banda estrecha (Narrowband)
            (6, 6),    # Bloque pequeño
            (8, 15),   # Banda ancha media
            (12, 30)   # Banda muy ancha (Wideband)
        ]
        
        for h, w in shapes:
            # Crear una plantilla con un "núcleo" de energía suave
            # Usamos un kernel gaussiano aproximado para que sea suave
            y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
            sigma = 0.3 * (h + w) / 2
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            
            # Normalizar plantilla (0 a 1)
            g = (g - g.min()) / (g.max() - g.min())
            templates.append(g)
            
        return templates

    def predict(self, spectrogram):
        """
        Aplica el banco de filtros sobre el espectrograma.
        """
        # Mapa acumulado de máximas correlaciones
        h, w = spectrogram.shape
        combined_score_map = np.zeros((h, w))
        
        # 1. Correlacionar cada plantilla
        for temp in self.templates:
            # match_template devuelve correlación (-1 a 1)
            # 'pad_input=True' mantiene el tamaño original
            res = match_template(spectrogram, temp, pad_input=True)
            
            # Nos quedamos con la máxima respuesta observada en cada pixel
            # (Max-Pooling a través de los filtros)
            combined_score_map = np.maximum(combined_score_map, res)
        
        # 2. Umbralizado
        mask = combined_score_map > self.threshold
        
        # 3. Extracción de cajas
        boxes = self._extract_boxes_from_mask(mask, combined_score_map)
        
        return {
            "mask": mask,
            "boxes": boxes,
            "threshold_map": combined_score_map # Para visualizar el "heatmap" de correlación
        }

    def _extract_boxes_from_mask(self, mask, score_map):
        """Convierte la máscara en cajas (similar a CFAR)."""
        labeled = label(mask)
        props = regionprops(labeled, intensity_image=score_map)
        
        boxes = []
        for prop in props:
            if prop.area < 5: continue # Ignorar ruido muy pequeño
            
            y1, x1, y2, x2 = prop.bbox
            score = prop.max_intensity
            
            boxes.append([x1, y1, x2, y2, score])
            
        return boxes