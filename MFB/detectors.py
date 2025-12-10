import numpy as np
from skimage.feature import match_template
from skimage.measure import label, regionprops

class MatchedFilterBankDetector:
    def __init__(self, threshold=0.55):
        """
        Detector Matched Filter Bank.
        threshold: Nivel de similitud (0.0 a 1.0).
        """
        self.threshold = threshold
        self.templates = self._build_template_bank()

    def _build_template_bank(self):
        """
        Crea moldes de señales sintéticas típicas en RF.
        """
        templates = []
        # Definimos (Alto, Ancho) de las señales esperadas
        # El sistema probará todas estas formas contra la imagen
        shapes = [
            (8, 3),   # Señal muy estrecha (Narrowband)
            (12, 5),  # Señal estrecha media
            (6, 6),   # Bloque pequeño
            (10, 15), # Señal ancha (Wideband media)
            (15, 30)  # Señal muy ancha (Wideband completa)
        ]
        
        for h, w in shapes:
            # Crear molde Gaussiano suave
            y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
            sigma = 0.35 * (h + w) / 2
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            # Normalizar molde
            g = (g - g.min()) / (g.max() - g.min())
            templates.append(g)
            
        return templates

    def predict(self, spectrogram):
        h, w = spectrogram.shape
        # Matriz para guardar la mejor coincidencia encontrada en cada pixel
        max_correlation_map = np.zeros((h, w))
        
        # 1. Probar cada filtro del banco
        for temp in self.templates:
            # Correlación normalizada
            res = match_template(spectrogram, temp, pad_input=True)
            # Max-Pooling: Nos quedamos con la mejor respuesta
            max_correlation_map = np.maximum(max_correlation_map, res)
        
        # 2. Umbralizar
        mask = max_correlation_map > self.threshold
        
        # 3. Extraer cajas
        boxes = self._extract_boxes(mask, max_correlation_map)
        
        return {
            "mask": mask,
            "boxes": boxes,
            "threshold_map": max_correlation_map
        }

    def _extract_boxes(self, mask, score_map):
        labeled = label(mask)
        props = regionprops(labeled, intensity_image=score_map)
        boxes = []
        
        for prop in props:
            # Filtro básico de ruido
            if prop.area < 6: continue 
            
            y1, x1, y2, x2 = prop.bbox
            score = prop.max_intensity
            boxes.append([x1, y1, x2, y2, score])
            
        return boxes