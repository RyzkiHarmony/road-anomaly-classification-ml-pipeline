import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

_PROJECT_ROOT = r'd:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines'
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "cnn_model"))

from config import CNN_OUT_DIR, get_logger
from model import InceptionTime1D
from train import DynamicJitterDataset, scale_instance_level, get_stratified_group_split

logger = get_logger(__name__)

def evaluate_fold(model, loader, device, p_idx, sb_idx, ne_idx):
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

    preds = np.argmax(probas, axis=1)
    
    pothole_f1 = f1_score(trues, preds, labels=[p_idx], average="macro", zero_division=0)
    sb_f1 = f1_score(trues, preds, labels=[sb_idx], average="macro", zero_division=0) if sb_idx != -1 else 0.0
    return pothole_f1, sb_f1

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_path = os.path.join(CNN_OUT_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(CNN_OUT_DIR, "cnn_1d_y.npy")
    g_path = os.path.join(CNN_OUT_DIR, "cnn_1d_groups.npy")

    X = np.load(X_path)
    y_raw = np.load(y_path)
    groups = np.load(g_path)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    ne_idx = list(classes).index("Non-Event")

    dev_groups_list, _ = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev = X[dev_mask]
    y_dev = y[dev_mask]
    groups_dev = groups[dev_mask]

    X_dev = scale_instance_level(X_dev)

    lrs = [0.0005, 0.001]
    dropouts = [0.2, 0.5]
    
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    best_pothole_f1 = -1.0
    best_params = None

    logger.info("Starting CNN Hyperparameter Sweep (LR, Dropout) for InceptionTime1D...")

    for lr in lrs:
        for drop in dropouts:
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
                X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
                X_vl, y_vl = X_dev[val_idx], y_dev[val_idx]

                tr_ds = DynamicJitterDataset(
                    torch.tensor(X_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.long),
                    is_train=True, non_event_idx=ne_idx
                )
                vl_ds = DynamicJitterDataset(
                    torch.tensor(X_vl, dtype=torch.float32),
                    torch.tensor(y_vl, dtype=torch.long),
                    is_train=False, non_event_idx=ne_idx, max_jitter=0
                )

                tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
                vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False)

                cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
                cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

                model = InceptionTime1D(in_channels=X_tr.shape[1], num_classes=len(classes), dropout_rate=drop).to(device)
                criterion = nn.CrossEntropyLoss(weight=cw_t)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                for epoch in range(15):
                    model.train()
                    train_loss = 0.0
                    for bx, by in tr_ld:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * bx.size(0)
                    train_loss /= len(tr_ld.dataset)
                    logger.info(f"    Fold {fold+1} Epoch {epoch+1}/15 Train Loss: {train_loss:.4f}")

                p_f1, sb_f1 = evaluate_fold(model, vl_ld, device, p_idx, sb_idx, ne_idx)
                fold_scores.append((p_f1, sb_f1))

            avg_p_f1 = np.mean([s[0] for s in fold_scores])
            avg_sb_f1 = np.mean([s[1] for s in fold_scores])

            logger.info(f"Params: LR={lr:.4f}, Dropout={drop:.1f} | OOF Pothole F1: {avg_p_f1:.4f} | SB F1: {avg_sb_f1:.4f}")

            if avg_p_f1 > best_pothole_f1:
                best_pothole_f1 = avg_p_f1
                best_params = (lr, drop)

    print("\n" + "=" * 60)
    print("                 BEST 1D-CNN HYPERPARAMETERS                 ")
    print("=" * 60)
    print(f"Best Params (LR, Dropout): {best_params}")
    print(f"Best OOF Pothole F1: {best_pothole_f1:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
