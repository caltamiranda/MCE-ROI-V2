import torch
import numpy as np
import time
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score

# --- Setup de Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Imports locales
import config as cfg
from core.data_loader import H5HybridDetectionDataset, collate_eval 
from models.hybrid_classifier import HybridRFClassifier
from core.roi_detection import ROIDetector
from core.feature_engineering import FeatureEngineer
from core.visual_stream import VisualStream
from core.detection_metrics import compute_map


# =============================================================================
# Función para crear carpeta de resultados organizada
# =============================================================================
def create_results_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"experiment_{timestamp}"
    results_path = os.path.join(current_dir, "results", folder_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"[Folder] Resultados guardados en: {results_path}")
    return results_path


# =============================================================================
# VALIDACIÓN (DETECCIÓN FULL SPECTROGRAM)
# =============================================================================
def validate_full_detection(model, val_loader, detector, fe, vs, device): 
    model.eval()
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            S_mce = sample['S_mce']
            S_det = sample['S_det']
            iq = sample['iq']
            gt = sample['gt_boxes']
            
            rois = detector.detect(S_det)

            if len(rois) == 0:
                all_pred_boxes.append(np.array([]))
                all_pred_scores.append(np.array([]))
                all_gt_boxes.append(gt)
                continue

            roi_feats = fe.features_for_rois(iq, S_det, rois)
            feat_map = {(r.roi.y1, r.roi.x1, r.roi.y2, r.roi.x2): r.features for r in roi_feats}
            
            patches = vs.extract_patches(S_mce, rois)

            X_vis, X_eng, box_coords = [], [], []
            for k, r in enumerate(rois):
                key = (r.y1, r.x1, r.y2, r.x2)
                if key in feat_map and k < len(patches) and patches[k] is not None:
                    X_eng.append(torch.tensor(feat_map[key], dtype=torch.float32))
                    X_vis.append(torch.tensor(patches[k], dtype=torch.float32))
                    box_coords.append([r.x1, r.y1, r.x2, r.y2])

            if not X_vis:
                all_pred_boxes.append(np.array([]))
                all_pred_scores.append(np.array([]))
                all_gt_boxes.append(gt)
                continue

            X_vis = torch.stack(X_vis).to(device)
            X_eng = torch.stack(X_eng).to(device)
            
            logits = model(X_vis, X_eng)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_pred_boxes.append(np.array(box_coords))
            all_pred_scores.append(probs)
            all_gt_boxes.append(gt)

    map50 = compute_map(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold=0.5)
    return map50


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def main():
    print(f"[Init] Dataset Train: {cfg.TRAIN_H5}")
    print(f"[Init] Dataset Val:   {cfg.VAL_H5}")

    # Crear carpeta para guardar todo
    results_path = create_results_folder()

    # 1. DataLoaders
    ds_train = H5HybridDetectionDataset(cfg.TRAIN_H5, mode="train")
    ds_val   = H5HybridDetectionDataset(cfg.VAL_H5, mode="eval")

    dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_eval)

    # 2. Modelo
    model = HybridRFClassifier(
        num_classes=2, 
        feat_dim=32,
        img_in_ch=3,
        cnn_channels=(16, 32, 64),
        mlp_hidden=(64, 64)
    ).to(cfg.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Pipeline Auxiliar
    detector = ROIDetector(
        method="zscore",
        z_thresh=4.5,
        background_window=65,
        gauss_sigma=(0.0, 3.0),
        merge_dist=3,
        min_area=60,
        min_ar=0.3,
        max_ar=20.0,
    )
    
    fe = FeatureEngineer(fs=cfg.FS, nperseg=cfg.NPERSEG, noverlap=cfg.NOVERLAP)
    vs = VisualStream(target_size=cfg.IMG_SIZE)

    # 4. Historial de Métricas
    history = {"train_loss": [], "train_acc": [], "train_recall": [], "val_map": []}
    
    best_map = 0.0
    print("[Start] Iniciando entrenamiento...")

    for epoch in range(cfg.EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.EPOCHS} ===")
        
        model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        
        start_train = time.time()
        for i, (x_vis, x_eng, y) in enumerate(dl_train):
            x_vis, x_eng, y = x_vis.to(cfg.DEVICE), x_eng.to(cfg.DEVICE), y.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            logits = model(x_vis, x_eng)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
        
        # Métricas Train
        avg_loss = epoch_loss / len(dl_train)
        train_acc = accuracy_score(all_targets, all_preds)
        train_rec = recall_score(all_targets, all_preds, zero_division=0)

        # Validación
        map_score = validate_full_detection(model, dl_val, detector, fe, vs, cfg.DEVICE)

        print(f"   [Train] Loss: {avg_loss:.4f} | Acc: {train_acc*100:.2f}% | Recall: {train_rec*100:.2f}%")
        print(f"   [Val] mAP@0.5: {map_score:.4f}")

        # Guardar historial en carpeta
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["train_recall"].append(train_rec)
        history["val_map"].append(map_score)
        
        np.save(os.path.join(results_path, "history_loss.npy"), np.array(history["train_loss"]))
        np.save(os.path.join(results_path, "history_acc.npy"), np.array(history["train_acc"]))
        np.save(os.path.join(results_path, "history_recall.npy"), np.array(history["train_recall"]))
        np.save(os.path.join(results_path, "history_map.npy"), np.array(history["val_map"]))

        # Guardar modelos
        torch.save(model.state_dict(), os.path.join(results_path, "last_model.pt"))
        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), os.path.join(results_path, "best_hybrid_detector.pt"))
            print(f"   [SAVE] 🏆 Nuevo mejor modelo guardado (mAP: {best_map:.4f})")

if __name__ == "__main__":
    main()
