import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk

class CFARDetector2D:
    def __init__(self, num_train=4, num_guard=8, p_fa=0.01, min_w=3, min_h=3, min_area=10):
        """
        Detector CFAR con filtros de tamaño y limpieza morfológica.
        
        Args:
            min_w (int): Ancho mínimo en píxeles.
            min_h (int): Alto mínimo en píxeles.
            min_area (int): Área mínima (número de píxeles activos).
        """
        self.Tc = num_train
        self.Gc = num_guard
        self.Pfa = p_fa
        
        # Filtros de tamaño
        self.min_w = min_w
        self.min_h = min_h
        self.min_area = min_area
        
        self.kernel, self.N_train = self._build_kernel()
        self.alpha = self.N_train * (self.Pfa**(-1/self.N_train) - 1)

    def _build_kernel(self):
        win_size = 1 + 2 * (self.Tc + self.Gc)
        kernel = np.ones((win_size, win_size))
        center = self.Tc + self.Gc
        start_guard = center - self.Gc
        end_guard = center + self.Gc + 1
        kernel[start_guard:end_guard, start_guard:end_guard] = 0
        return kernel, np.sum(kernel)

    def predict(self, spectrogram):
        # 1. CFAR Estándar
        noise_sum = convolve(spectrogram, self.kernel, mode='mirror')
        noise_level = noise_sum / self.N_train
        threshold_map = noise_level * self.alpha
        mask = spectrogram > threshold_map
        
        # 2. LIMPIEZA MORFOLÓGICA (Nuevo)
        # 'binary_opening' elimina puntos blancos pequeños (ruido sal y pimienta)
        # Usamos un disco pequeño (radio 1) para borrar píxeles aislados
        clean_mask = binary_opening(mask, footprint=disk(1))
        
        # 3. Extracción con filtros
        boxes = self._extract_boxes(clean_mask, spectrogram)
        
        return {
            "mask": clean_mask, # Devolvemos la máscara limpia
            "boxes": boxes,
            "threshold_map": threshold_map
        }

    def _extract_boxes(self, mask, spectrogram):
        labeled_mask = label(mask)
        props = regionprops(labeled_mask, intensity_image=spectrogram)
        
        boxes = []
        for prop in props:
            # coords: (min_row, min_col, max_row, max_col) -> (y1, x1, y2, x2)
            y1, x1, y2, x2 = prop.bbox
            
            width = x2 - x1
            height = y2 - y1
            
            # --- FILTROS ESTRICTOS ---
            # 1. Filtro de Área (pixeles totales)
            if prop.area < self.min_area:
                continue
                
            # 2. Filtro de Dimensiones (evitar lineas de 1px de ancho)
            if width < self.min_w or height < self.min_h:
                continue
            
            # Si pasa los filtros, guardar
            score = prop.max_intensity
            boxes.append([x1, y1, x2, y2, score])
            
        return boxes