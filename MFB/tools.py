import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_spectrogram_mfb(path):
    """
    Carga el espectrograma con corrección Gamma para MFB.
    El MFB necesita un contraste alto para que la correlación funcione bien.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró: {path}")

    try:
        img = Image.open(path).convert('RGB')
        _, g, _ = img.split() # Usamos canal verde
        
        data = np.array(g).astype(np.float32)
        
        # 1. Normalizar 0-1
        val_min, val_max = data.min(), data.max()
        if val_max > val_min:
            data = (data - val_min) / (val_max - val_min)
        
        # 2. Corrección Gamma (Importante para eliminar ruido de fondo)
        # Esto hace que el fondo gris se vuelva negro
        data = data ** 4.0 
        
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return np.zeros((64, 64))
        
    return data

def load_yolo_labels(path, img_width, img_height):
    """Carga etiquetas Ground Truth."""
    if not os.path.exists(path): return []
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            try:
                cls, xc, yc, w, h = map(float, parts[:5])
                x1 = (xc - w/2) * img_width
                y1 = (yc - h/2) * img_height
                x2 = (xc + w/2) * img_width
                y2 = (yc + h/2) * img_height
                boxes.append([x1, y1, x2, y2, cls])
            except: continue
    return boxes

def visualize_mfb(spectrogram, result, gt_boxes=None, output_path=None):
    """Visualiza la entrada, el mapa de correlación y la detección final."""
    score_map = result['threshold_map'] # Mapa de calor de correlación
    pred_boxes = result['boxes']
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Entrada
    ax[0].imshow(spectrogram, aspect='auto', cmap='inferno')
    ax[0].set_title("Input (Gamma Corrected)")
    if gt_boxes:
        for b in gt_boxes:
            rect = plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], 
                                 edgecolor='#00FF00', fill=False, lw=1, ls='--')
            ax[0].add_patch(rect)

    # 2. Mapa de Correlación (Lo que "piensa" el filtro)
    im = ax[1].imshow(score_map, aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax[1].set_title("Mapa de Correlación (Similitud)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    # 3. Resultado Final
    ax[2].imshow(spectrogram, aspect='auto', cmap='gray')
    ax[2].set_title(f"Detección MFB [N={len(pred_boxes)}]")
    for box in pred_boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             edgecolor='red', fill=False, lw=2)
        ax[2].add_patch(rect)

    plt.tight_layout()
    if output_path: plt.savefig(output_path)
    plt.close() # Cerrar para no saturar memoria