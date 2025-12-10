import argparse
import os
import sys

# Truco para importar metrics.py desde la carpeta superior
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MFB.detectors import MatchedFilterBankDetector
from MFB.tools import load_spectrogram_mfb, load_yolo_labels, visualize_mfb
from metrics import DetectionEvaluator # Importado desde la raíz

def main():
    parser = argparse.ArgumentParser(description="Matched Filter Bank Benchmark")
    
    # Rutas (Ajusta a tus carpetas)
    default_imgs = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\images"
    default_lbls = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\labels"
    
    parser.add_argument('--input_dir', default=default_imgs)
    parser.add_argument('--label_dir', default=default_lbls)
    parser.add_argument('--output_dir', default='resultados_mfb')
    
    # Umbral de correlación (0.5 a 0.7 suele ser bueno)
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--max_plots', type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.png')])
    
    print(f"=== INICIO MFB (Matched Filter Bank) ===")
    print(f"Total imágenes: {len(image_files)}")
    print(f"Umbral Similitud: {args.threshold}")

    # Inicializar
    detector = MatchedFilterBankDetector(threshold=args.threshold)
    evaluator = DetectionEvaluator()
    plots_done = 0

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(args.input_dir, img_name)
        lbl_path = os.path.join(args.label_dir, img_name.replace('.png', '.txt'))
        
        try:
            # 1. Cargar
            spectrogram = load_spectrogram_mfb(img_path)
            h, w = spectrogram.shape
            
            gt_boxes = load_yolo_labels(lbl_path, w, h)
            
            # 2. Detectar
            results = detector.predict(spectrogram)
            
            # 3. Evaluar
            evaluator.update(img_name, results['boxes'], gt_boxes)
            
            # 4. Visualizar
            if plots_done < args.max_plots:
                out_path = os.path.join(args.output_dir, f"mfb_{img_name}")
                visualize_mfb(spectrogram, results, gt_boxes, out_path)
                plots_done += 1
            
            if idx % 50 == 0:
                print(f"[{idx}] {img_name} -> Detectados: {len(results['boxes'])}")
                
        except Exception as e:
            print(f"Error en {img_name}: {e}")

    print("\n=== RESULTADOS MFB ===")
    evaluator.print_summary()

if __name__ == "__main__":
    main()