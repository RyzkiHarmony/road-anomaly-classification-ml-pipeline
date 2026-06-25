# tune.py
# Skrip untuk pencarian hyperparameter CNN (LR dan Focal Loss gamma) tanpa SMOTE.

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))

from config import CNN_OUT_DIR, get_logger
from model import Lightweight1DCNN
from train import DynamicJitterDataset, FocalLoss, set_seed

logger = get_logger(__name__)

def evaluate_dataset(model, loader, device, p_idx, sb_idx, classes, best_thresh_p, best_thresh_sb, criterion):
    model.eval()
    probas, trues = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            out = model(bx)
            probs = torch.softmax(out, dim=1)
            probas.extend(probs.cpu().numpy())
            trues.extend(by.numpy())
    probas = np.array(probas)
    trues = np.array(trues)

    preds = np.zeros_like(trues)
    ne_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    for i, proba in enumerate(probas):
        pp = proba[p_idx]
        sbp = proba[sb_idx] if sb_idx != -1 else 0.0
        pt = pp >= best_thresh_p
        sbt = sb_idx != -1 and sbp >= best_thresh_sb
        if pt and sbt:
            preds[i] = p_idx if pp >= sbp else sb_idx
        elif pt:
            preds[i] = p_idx
        elif sbt:
            preds[i] = sb_idx
        else:
            preds[i] = ne_idx
    pothole_f1 = f1_score(trues, preds, labels=[p_idx], average="macro", zero_division=0)
    sb_f1 = f1_score(trues, preds, labels=[sb_idx], average="macro", zero_division=0) if sb_idx != -1 else 0.0
    return pothole_f1, sb_f1

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    X_path = os.path.join(CNN_OUT_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(CNN_OUT_DIR, "cnn_1d_y.npy")
    g_path = os.path.join(CNN_OUT_DIR, "cnn_1d_groups.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(g_path)):
        logger.error("Dataset files not found. Please run build_cnn_data.py first.")
        return

    X = np.load(X_path)
    y_raw = np.load(y_path)
    groups = np.load(g_path)

    # Label Encoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1

    # Dev/Split
    from train import get_stratified_group_split
    dev_groups_list, _ = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev = X[dev_mask]
    y_dev = y[dev_mask]
    groups_dev = groups[dev_mask]

    # Hyperparameter sweep grid (No SMOTE)
    lrs = [0.0005, 0.001, 0.002]
    gammas = [1.5, 2.0, 2.5]
    
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    best_composite = -1.0
    best_params = None

    logger.info("Starting SMOTE-free CNN parameter sweep over LR and Focal Loss Gamma...")

    for lr in lrs:
        for gamma in gammas:
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
                X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
                X_vl, y_vl = X_dev[val_idx], y_dev[val_idx]

                # Standardize
                ch_means = np.mean(X_tr, axis=(0, 2), keepdims=True)
                ch_stds = np.std(X_tr, axis=(0, 2), keepdims=True)
                ch_stds = np.where(ch_stds < 1e-6, 1.0, ch_stds)

                X_tr_norm = (X_tr - ch_means) / ch_stds
                X_vl_norm = (X_vl - ch_means) / ch_stds

                # Loader
                tr_ds = DynamicJitterDataset(
                    torch.tensor(X_tr_norm, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.long),
                    max_jitter=15, noise_std=0.02, scale_range=(0.85, 1.15), is_train=True
                )
                vl_ds = DynamicJitterDataset(
                    torch.tensor(X_vl_norm, dtype=torch.float32),
                    torch.tensor(y_vl, dtype=torch.long),
                    max_jitter=0, noise_std=0, scale_range=None, is_train=False
                )

                tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
                vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False)

                # Class Weights
                cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
                cw = np.sqrt(cw) / np.sqrt(cw).sum() * len(cw)
                cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

                model = Lightweight1DCNN(in_channels=14, num_classes=len(classes)).to(device)
                criterion = FocalLoss(weight=cw_t, gamma=gamma)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                # Simple training loop for tuning (15 epochs to speed up search)
                for epoch in range(15):
                    model.train()
                    for bx, by in tr_ld:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        optimizer.step()

                # Get optimal threshold for validation fold
                model.eval()
                val_probas = []
                with torch.no_grad():
                    for bx, _ in vl_ld:
                        bx = bx.to(device)
                        out = model(bx)
                        val_probas.extend(torch.softmax(out, dim=1).cpu().numpy())
                val_probas = np.array(val_probas)

                y_val_pothole = (y_vl == p_idx).astype(int)
                y_proba_pothole = val_probas[:, p_idx]
                prec, rec, thresholds = precision_recall_curve(y_val_pothole, y_proba_pothole)
                fscore = (2 * prec * rec) / (prec + rec + 1e-9)
                ix = np.argmax(fscore)
                best_t_p = thresholds[ix] if ix < len(thresholds) else 0.5

                best_t_sb = 0.5
                if sb_idx != -1:
                    y_val_sb = (y_vl == sb_idx).astype(int)
                    y_proba_sb = val_probas[:, sb_idx]
                    prec_sb, rec_sb, thresholds_sb = precision_recall_curve(y_val_sb, y_proba_sb)
                    fscore_sb = (2 * prec_sb * rec_sb) / (prec_sb + rec_sb + 1e-9)
                    ix_sb = np.argmax(fscore_sb)
                    best_t_sb = thresholds_sb[ix_sb] if ix_sb < len(thresholds_sb) else 0.5

                p_f1, sb_f1 = evaluate_dataset(model, vl_ld, device, p_idx, sb_idx, classes, best_t_p, best_t_sb, criterion)
                fold_scores.append((p_f1, sb_f1))

            avg_p_f1 = np.mean([s[0] for s in fold_scores])
            avg_sb_f1 = np.mean([s[1] for s in fold_scores])
            composite = 0.7 * avg_p_f1 + 0.3 * avg_sb_f1

            logger.info(f"Params: LR={lr:.4f}, Gamma={gamma:.1f} | Pothole F1: {avg_p_f1:.4f} | SB F1: {avg_sb_f1:.4f} | Composite: {composite:.4f}")

            if composite > best_composite:
                best_composite = composite
                best_params = (lr, gamma)

    print("\n" + "=" * 60)
    print("                 BEST 1D-CNN HYPERPARAMETERS                 ")
    print("=" * 60)
    print(f"Best Params (LR, Focal Loss Gamma): {best_params}")
    print(f"Best OOF Composite Score: {best_composite:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
