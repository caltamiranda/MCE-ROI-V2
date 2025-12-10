import numpy as np

def calculate_iou(box_a, box_b):
    """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

class DetectionEvaluator:
    def __init__(self):
        # Almacenamiento de datos
        self.all_preds = []  # Lista de dicts
        self.all_gts = {}    # Dict: img_id -> lista de boxes
        self.img_ids = []    # Para iterar en orden
        self.total_gt_boxes = 0

    def update(self, img_id, pred_boxes, gt_boxes):
        """
        Registra datos de una imagen.
        pred_boxes: [[x1, y1, x2, y2, score], ...]
        gt_boxes:   [[x1, y1, x2, y2, class], ...]
        """
        if img_id not in self.img_ids:
            self.img_ids.append(img_id)

        # Guardar GTs limpios (solo coords)
        gts_clean = [g[:4] for g in gt_boxes]
        self.all_gts[img_id] = gts_clean
        self.total_gt_boxes += len(gts_clean)

        # Guardar Predicciones
        for p in pred_boxes:
            self.all_preds.append({
                'img_id': img_id,
                'bbox': p[:4],
                'score': p[4]
            })

    # --- PARTE 1: PRECISION, RECALL, F1 (Escalares) ---
    def compute_basic_metrics(self, iou_threshold=0.5):
        """
        Calcula P, R, F1 basándose en las cajas detectadas actualmente,
        sin variar umbrales de confianza. Ideal para CFAR.
        """
        tp_total = 0
        fp_total = 0
        fn_total = 0

        # Iterar imagen por imagen
        for img_id in self.img_ids:
            gts = self.all_gts[img_id]
            # Filtrar predicciones de esta imagen
            preds = [p['bbox'] for p in self.all_preds if p['img_id'] == img_id]

            # Contadores locales
            matched_gt = set()
            
            # Ordenar preds por si acaso (aunque en métricas básicas no es estricto)
            # Aquí asumimos que todas las preds son "positivas"
            
            # Matriz de IoU
            if len(gts) == 0:
                fp_total += len(preds)
                continue
            
            if len(preds) == 0:
                fn_total += len(gts)
                continue

            # Greedy Matching
            # Iteramos sobre predicciones y buscamos su mejor GT
            for p_idx, pred_box in enumerate(preds):
                best_iou = -1
                best_gt_idx = -1
                
                for g_idx, gt_box in enumerate(gts):
                    if g_idx in matched_gt: continue # Ya fue emparejado
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= iou_threshold:
                    tp_total += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp_total += 1
            
            # Los GT que no se emparejaron son False Negatives
            fn_total += (len(gts) - len(matched_gt))

        # Cálculos finales
        precision = tp_total / (tp_total + fp_total + 1e-6)
        recall = tp_total / (tp_total + fn_total + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return {
            "TP": tp_total, "FP": fp_total, "FN": fn_total,
            "Precision": precision, "Recall": recall, "F1": f1
        }

    # --- PARTE 2: mAP (Mean Average Precision) ---
    def compute_ap_at_iou(self, iou_threshold):
        """Calcula AP para un IoU dado ordenando por score."""
        if self.total_gt_boxes == 0 or len(self.all_preds) == 0:
            return 0.0

        sorted_preds = sorted(self.all_preds, key=lambda x: x['score'], reverse=True)
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))
        
        gt_matched = {img_id: np.zeros(len(boxes), dtype=bool) 
                      for img_id, boxes in self.all_gts.items()}

        for i, pred in enumerate(sorted_preds):
            img_id = pred['img_id']
            pred_box = pred['bbox']
            gt_boxes = self.all_gts[img_id]

            best_iou = -1
            best_gt_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold:
                if not gt_matched[img_id][best_gt_idx]:
                    tp[i] = 1
                    gt_matched[img_id][best_gt_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / self.total_gt_boxes
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap

    def print_summary(self):
        # 1. Calcular Métricas Básicas (P, R, F1)
        basic = self.compute_basic_metrics(iou_threshold=0.5)
        
        # 2. Calcular mAP
        map50 = self.compute_ap_at_iou(0.50)
        thresholds = np.linspace(0.5, 0.95, 10)
        aps = [self.compute_ap_at_iou(t) for t in thresholds]
        map_coco = np.mean(aps)

        print("\n" + "="*50)
        print(f"       RESULTADOS DE EVALUACIÓN (Total GT: {self.total_gt_boxes})")
        print("="*50)
        print(" [ MÉTRICAS BÁSICAS (IoU=0.5) ]")
        print(f"  > True Positives (TP)  : {basic['TP']}")
        print(f"  > False Positives (FP) : {basic['FP']}")
        print(f"  > False Negatives (FN) : {basic['FN']}")
        print("-" * 30)
        print(f"  > PRECISION            : {basic['Precision']:.4f}")
        print(f"  > RECALL               : {basic['Recall']:.4f}")
        print(f"  > F1-SCORE             : {basic['F1']:.4f}")
        print("="*50)
        print(" [ MÉTRICAS AVANZADAS (mAP) ]")
        print(f"  > mAP @ IoU=0.50       : {map50:.4f}")
        print(f"  > mAP @ IoU=0.50:0.95  : {map_coco:.4f}")
        print("="*50 + "\n")