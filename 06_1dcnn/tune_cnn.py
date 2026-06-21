import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
import optuna

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_pipeline_experiment'))
from config import get_logger

from model import Lightweight1DCNN
from train_cnn import DynamicJitterDataset, FocalLoss, apply_smote, SMOTE_RATIO

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

EPOCHS = 30

def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    conv1_filters = trial.suggest_categorical("conv1_filters", [16, 32])
    conv2_filters = trial.suggest_categorical("conv2_filters", [32, 64])
    
    # Load Data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))
    groups = np.load(os.path.join(DATA_DIR, "groups.npy"))
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3-Fold CV for speed during tuning
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_pr_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        # Apply SMOTE only on training fold
        X_train_np, y_train_np = X[train_idx], y[train_idx]
        X_train_smote, y_train_smote = apply_smote(X_train_np, y_train_np, ratio=SMOTE_RATIO)
        
        X_train = torch.tensor(X_train_smote, dtype=torch.float32)
        y_train = torch.tensor(y_train_smote, dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)
        
        # Datasets with training augmentations
        train_dataset = DynamicJitterDataset(X_train, y_train, max_jitter=15, 
                                              noise_std=0.02, scale_range=(0.85, 1.15), is_train=True)
        val_dataset = DynamicJitterDataset(X_val, y_val, max_jitter=0,
                                            noise_std=0, scale_range=None, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Dampened Class Weights (sqrt)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
        class_weights = np.sqrt(class_weights)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # 10 Channels to match the actual dataset features
        model = Lightweight1DCNN(in_channels=10, num_classes=len(classes), 
                                 conv1_filters=conv1_filters, conv2_filters=conv2_filters, dropout_rate=dropout_rate).to(device)
        
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
                
            # Eval
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                
        # Eval Best Model for this fold
        model.load_state_dict(best_model_state)
        model.eval()
        all_probas = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                all_probas.extend(probs.cpu().numpy())
                all_trues.extend(batch_y.numpy())
                
        all_probas = np.array(all_probas)
        all_trues = np.array(all_trues)
        
        y_test_pothole = (all_trues == p_idx).astype(int)
        y_proba_pothole = all_probas[:, p_idx]
        prec, rec, _ = precision_recall_curve(y_test_pothole, y_proba_pothole)
        pr_auc_val = auc(rec, prec)
        
        fold_pr_aucs.append(pr_auc_val)
        
    avg_prauc = np.mean(fold_pr_aucs)
    return avg_prauc

def main():
    logger.info("Memulai Optuna Study untuk 1D-CNN (Max 20 Trials)")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    logger.info("Tuning Selesai!")
    logger.info("Best Trial:")
    logger.info(f"  PR-AUC: {study.best_value:.4f}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

if __name__ == "__main__":
    main()
