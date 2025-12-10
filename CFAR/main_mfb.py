import argparse
import os
import sys
import numpy as np

# --- IMPORTS ---
from detectors.cfar import CFARDetector2D
from detectors.mfb import MatchedFilterBankDetector # <--- IMPORTAR EL NUEVO
from utils.tools import load_spectrogram, load_yolo_labels, visualize_results
from metrics import DetectionEvaluator

def main():
    parser = argparse.ArgumentParser(description="Wideband Detection Benchmark")
    
    # Rutas (Ajusta las tuyas)
    default_imgs = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\images"
    default_lbls = r"C:\Users\User\Documents\GitHub\MCE-ROI-V2\rf_benchmark\spectrograms_yolo\test\labels"
    
    parser.add_argument('--input_dir', type=str, default=default_imgs)
    parser.add_argument('--label_dir', type=str, default=default_lbls)
    parser.add_argument('--output_dir', type=str, default='resultados_benchmark')
    
    # --- SELECCIÓN DE DETECTOR ---
    parser.add_argument('--method', type=str, default='cfar', choices=['cfar', 'mfb'],
                        help="Algoritmo a usar: 'cfar' o 'mfb' (Matched Filter)")

    # Parametros CFAR
    parser.add_argument('--pfa', type=float, default=0.01)
    
    # Parametros Matched Filter
    parser.add_argument('--mfb_thresh', type=float, default=0.6, 
                        help="Umbral de similitud para MFB (0.0 a 1.0)")
    
    parser.add_argument('--max_plots', type=int, default=5)

    args = parser.parse_args()

    # ... (Validaciones de directorios igual que antes) ...
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]
    image_files.sort()

    print(f"=== INICIO BENCHMARK: {args.method.upper()} ===")

    # 1. Instanciar el Detector seleccionado
    if args.method == 'cfar':
        print(f"Configurando CFAR (Pfa={args.pfa})...")
        detector = CFARDetector2D(num_train=4, num_guard=8, p_fa=args.pfa)
    elif args.method == 'mfb':
        print(f"Configurando Matched Filter Bank (Thresh={args.mfb_thresh})...")
        detector = MatchedFilterBankDetector(threshold=args.mfb_thresh)

    evaluator = DetectionEvaluator()
    plots_done = 0
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(args.input_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(args.label_dir, txt_name)
        
        try:
            spectrogram = load_spectrogram(img_path)
            h, w = spectrogram.shape
            
            gt_boxes = []
            if os.path.exists(lbl_path):
                gt_boxes = load_yolo_labels(lbl_path, w, h)
            
            # Inferencia Polimórfica (funciona igual para ambos detectores)
            results = detector.predict(spectrogram)
            
            evaluator.update(img_name, results['boxes'], gt_boxes)
            
            if idx % 50 == 0:
                print(f"[{idx}] {img_name}: {len(results['boxes'])} dets")

            if plots_done < args.max_plots:
                out_path = os.path.join(args.output_dir, f"{args.method}_{img_name}")
                visualize_results(spectrogram, results, gt_boxes=gt_boxes, output_path=out_path)
                plots_done += 1
                
        except Exception as e:
            print(f"Error: {e}")

    print("\n=== RESULTADOS FINALES ===")
    evaluator.print_summary()

if __name__ == "__main__":
    main()