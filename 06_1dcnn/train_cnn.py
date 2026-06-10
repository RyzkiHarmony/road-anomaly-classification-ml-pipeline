import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_recall_curve, auc

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_pipeline_experiment'))
from config import get_logger

from model import Lightweight1DCNN

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 64
LR = 0.000828659730834538

def main():
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    groups_path = os.path.join(DATA_DIR, "groups.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(groups_path)):
        logger.error("Data tidak ditemukan. Jalankan build_dataset_cnn.py terlebih dahulu.")
        return
        
    X = np.load(X_path)
    y_raw = np.load(y_path)
    groups = np.load(groups_path)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"=== Fold {fold+1} ===")
        
        X_train, y_train = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.long)
        X_val, y_val = torch.tensor(X[val_idx], dtype=torch.float32), torch.tensor(y[val_idx], dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Hitung class weights untuk loss function (meredam imbalanced)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        model = Lightweight1DCNN(in_channels=3, num_classes=len(classes),
                                 conv1_filters=32, conv2_filters=64, dropout_rate=0.169).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_probas = []
            val_trues = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_probas.extend(probs.cpu().numpy())
                    val_trues.extend(batch_y.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        
        # Calculate final fold metrics
        model.eval()
        all_probas = []
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_probas.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(batch_y.numpy())
                
        all_probas = np.array(all_probas)
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        f1_val = f1_score(all_trues, all_preds, labels=[p_idx], average='macro', zero_division=0)
        
        # PR-AUC for Pothole
        y_test_pothole = (all_trues == p_idx).astype(int)
        y_proba_pothole = all_probas[:, p_idx]
        prec, rec, _ = precision_recall_curve(y_test_pothole, y_proba_pothole)
        pr_auc_val = auc(rec, prec)
        
        fold_metrics.append((f1_val, pr_auc_val))
        logger.info(f"Fold {fold+1} Best Val Loss: {best_val_loss:.4f} | Val Pothole F1: {f1_val:.4f} | Val PR-AUC: {pr_auc_val:.4f}")
        
        oof_y_true.extend(all_trues)
        oof_y_pred.extend(all_preds)
        oof_y_proba.extend(all_probas)
        
    avg_f1 = np.mean([m[0] for m in fold_metrics])
    avg_prauc = np.mean([m[1] for m in fold_metrics])
    print("-" * 60)
    logger.info(f"Average Pothole F1: {avg_f1:.4f} | Average PR-AUC: {avg_prauc:.4f}")
    
    # --- Cetak Classification Report ---
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)
    
    from sklearn.metrics import classification_report
    
    # Threshold Optimization for Pothole
    y_true_pothole = (oof_y_true == p_idx).astype(int)
    y_proba_pothole = oof_y_proba[:, p_idx]
    prec, rec, thresholds = precision_recall_curve(y_true_pothole, y_proba_pothole)
    
    # Cari threshold yang memaksimalkan F1-Score
    fscore = (2 * prec * rec) / (prec + rec + 1e-9)
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix] if ix < len(thresholds) else 0.5
    
    print("\n" + "=" * 60)
    print("              THRESHOLD OPTIMIZATION (Pothole)              ")
    print("=" * 60)
    print(f"Proposed Threshold for Pothole: {best_thresh:.4f}")
    print(f"Expected -> Precision: {prec[ix]:.4f}, Recall: {rec[ix]:.4f}, F1: {fscore[ix]:.4f}")
    print("\nFinal Report (Out-of-Fold - Unbiased Original):")
    print(classification_report(oof_y_true, oof_y_pred, target_names=classes))
    print("=" * 60 + "\n")
    
    # --- Train Final Model ---
    logger.info("Melatih final model dengan seluruh dataset...")
    X_full = torch.tensor(X, dtype=torch.float32)
    y_full = torch.tensor(y, dtype=torch.long)
    full_dataset = TensorDataset(X_full, y_full)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_weights_full = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_full = torch.tensor(class_weights_full, dtype=torch.float32).to(device)
    
    final_model = Lightweight1DCNN(in_channels=3, num_classes=len(classes),
                                   conv1_filters=32, conv2_filters=64, dropout_rate=0.169).to(device)
    criterion_full = nn.CrossEntropyLoss(weight=class_weights_full)
    optimizer_full = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        final_model.train()
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_full.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion_full(outputs, batch_y)
            loss.backward()
            optimizer_full.step()
            
    # Save Model
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, "best_1dcnn.pth"))
    
    # Save Label Encoder Classes
    np.save(os.path.join(MODEL_DIR, "classes.npy"), classes)
    
    logger.info("Model final PyTorch disimpan di " + os.path.join(MODEL_DIR, "best_1dcnn.pth"))
    
if __name__ == "__main__":
    main()
