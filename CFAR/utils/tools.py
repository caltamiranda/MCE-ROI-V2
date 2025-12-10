import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os

def load_spectrogram(path):
    """
    Carga el espectrograma aplicando corrección Gamma para limpiar el ruido.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    try:
        img = Image.open(path).convert('RGB')
        r, g, b = img.split()
        
        # El canal VERDE (g) suele tener la mejor relación señal-ruido en mapas 'inferno'
        data = np.array(g).astype(np.float32)
        
        # 1. Normalización Min-Max estándar
        val_min = data.min()
        val_max = data.max()
        if val_max - val_min > 0:
            data = (data - val_min) / (val_max - val_min)
        
        # 2. CORRECCIÓN GAMMA (La clave para tu problema)
        # Elevar a la potencia 4 hace que los grises medios (0.5) bajen a (0.06).
        # El ruido de fondo desaparece, la señal fuerte (0.9) se mantiene alta (0.65).
        data = data ** 4.0 
        
        # 3. Limpieza final opcional (para quitar ruido "pimienta" muy bajo)
        data[data < 0.1] = 0.0
            
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return np.zeros((64, 64))
        
    return data

def load_yolo_labels(path, img_width, img_height):
    """Carga etiquetas formato YOLO."""
    if not os.path.exists(path):
        return []

    boxes = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            try:
                class_id = int(parts[0])
                # Leer normalizados
                x_c, y_c, w, h = map(float, parts[1:5])
                
                # Convertir a pixeles
                w_px = w * img_width
                h_px = h * img_height
                x_c_px = x_c * img_width
                y_c_px = y_c * img_height
                
                x1 = x_c_px - (w_px / 2)
                y1 = y_c_px - (h_px / 2)
                x2 = x_c_px + (w_px / 2)
                y2 = y_c_px + (h_px / 2)
                
                boxes.append([x1, y1, x2, y2, class_id])
            except ValueError:
                continue
    return boxes

def visualize_results(spectrogram, result, gt_boxes=None, output_path=None):
    """Visualiza resultados."""
    mask = result['mask']
    pred_boxes = result['boxes']
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Input (Lo que ve la red realmente, o sea el canal verde procesado)
    ax[0].imshow(spectrogram, aspect='auto', cmap='inferno', origin='upper')
    ax[0].set_title(f"Input Procesado (Fondo limpio) + GT [N={len(gt_boxes) if gt_boxes else 0}]")
    
    if gt_boxes:
        for box in gt_boxes:
            x1, y1, x2, y2, cls_id = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 edgecolor='#00FF00', facecolor='none', linewidth=1.5, linestyle='--')
            ax[0].add_patch(rect)

    # Panel 2: Detección
    ax[1].imshow(spectrogram, aspect='auto', cmap='gray', origin='upper')
    
    for box in pred_boxes:
        x1, y1, x2, y2, score = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             edgecolor='red', facecolor='none', linewidth=2)
        ax[1].add_patch(rect)
    
    ax[1].set_title(f"Detección CFAR [N={len(pred_boxes)}]")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()