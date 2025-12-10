import argparse
import os
import sys
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops

# --- GESTIÓN DE IMPORTACIONES ---
# Añadimos la carpeta raíz al path para poder ver los otros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Importaciones del propio módulo actual
from RAW_CNN.model import Spectrogram1DCNN
from RAW_CNN.dataset import SpectrogramDataset

# 2. Importaciones de métricas globales
from metrics import DetectionEvaluator

# 3. Importaciones de herramientas compartidas (Asegúrate de que estas rutas existan)
# Si 'tools.py' está directamente en 'CFAR', quita '.utils'
try:
    from CFAR.utils.tools import visualize_results 
except ImportError:
    # Fallback por si la carpeta es diferente
    from CFAR.utils.tools import visualize_results 

try:
    from MFB.tools import load_yolo_labels
except ImportError:
    # Definición local si falla la importación de MFB
    def load_yolo_labels(path, w, h): return []


# --- FUNCIÓN CRÍTICA PARA EVITAR EL ERROR DE BATCH ---
def custom_collate_fn(batch):
    """
    Permite agrupar imágenes que tienen diferente número de cajas (Ground Truth).
    Sin esto, el DataLoader explota.
    """
    imgs = []
    masks = []
    boxes = []
    names = []
    
    for item in batch:
        imgs.append(item[0])
        masks.append(item[1])
        boxes.append(item[2]) # Mantenemos cajas como lista de listas
        names.append(item[3])
        
    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    
    return imgs, masks, boxes, names

def extract_boxes_from_heatmap(heatmap, threshold=0.5):
    """Convierte la salida de la CNN (mapa de probabilidad) en cajas."""
    mask = heatmap > threshold
    lbl = label(mask)
    props = regionprops(lbl, intensity_image=heatmap)
    boxes = []
    for prop in props:
        if prop.area < 5: continue
        y1, x1, y2, x2 = prop.bbox
        score = prop.max_intensity
        boxes.append([x1, y1, x2, y2, score])
    return boxes, mask

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # Desempaquetamos 4 valores gracias a custom_collate_fn
    for imgs, masks, _, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        output_map, _, _ = model(imgs)
        
        loss = criterion(output_map.unsqueeze(1), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    # Rutas por defecto
    default_imgs = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\images"
    default_lbls = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\labels"
    
    parser.add_argument('--input_dir', default=default_imgs)
    parser.add_argument('--label_dir', default=default_lbls)
    parser.add_argument('--output_dir', default='resultados_cnn')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_plots', type=int, default=5)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== RAW-IQ 1D CNN SIMULATOR (Device: {device}) ===")

    # 1. Cargar Datos
    dataset = SpectrogramDataset(args.input_dir, args.label_dir)
    
    # IMPORTANTE: Añadir collate_fn aquí
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    
    # 2. Inicializar Modelo
    model = Spectrogram1DCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    # 3. ENTRENAMIENTO
    print(f"Entrenando por {args.epochs} épocas...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, criterion, device)
        if (epoch+1) % 10 == 0: # Imprimir cada 10 épocas para no saturar
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

    # 4. INFERENCIA Y EVALUACIÓN
    print("\nIniciando Evaluación...")
    model.eval()
    evaluator = DetectionEvaluator()
    plots_done = 0
    
    # IMPORTANTE: Añadir collate_fn aquí también
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    with torch.no_grad():
        for imgs, _, _, img_names in test_loader:
            imgs = imgs.to(device)
            img_name = img_names[0]
            
            # Predicción
            output_map, _, _ = model(imgs)
            heatmap = output_map.cpu().numpy()[0]
            
            # Extraer Cajas
            pred_boxes, mask = extract_boxes_from_heatmap(heatmap, threshold=0.5)
            
            # Cargar Ground Truth
            lbl_path = os.path.join(args.label_dir, img_name.replace('.png', '.txt'))
            real_gt_boxes = load_yolo_labels(lbl_path, 64, 64)

            # Métricas
            evaluator.update(img_name, pred_boxes, real_gt_boxes)

            # Visualizar
            if plots_done < args.max_plots:
                result = {'mask': mask, 'boxes': pred_boxes}
                spec_vis = imgs.cpu().numpy()[0, 0] 
                out_path = os.path.join(args.output_dir, f"cnn_{img_name}")
                visualize_results(spec_vis, result, gt_boxes=real_gt_boxes, output_path=out_path)
                plots_done += 1

    print("\n=== RESULTADOS CNN ===")
    evaluator.print_summary()

if __name__ == "__main__":
    main()