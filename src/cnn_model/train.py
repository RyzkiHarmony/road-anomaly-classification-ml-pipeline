import os
import argparse
import random
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_recall_curve, auc, precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE

import sys
# Fix Windows Emoji crash in PyTorch ONNX exporter
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))

from config import CNN_OUT_DIR, get_logger
from model import InceptionTime1D
from data_utils import get_stratified_group_split

logger = get_logger(__name__)

DATA_DIR = CNN_OUT_DIR
MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d")
REPORT_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "cnn_1d")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


EPOCHS = 40
BATCH_SIZE = 16
LR = 0.0007795
SMOTE_RATIO = 0.5

class DynamicJitterDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_jitter=15, noise_std=0.02, scale_range=(0.85, 1.15),
                 time_warp_prob=0.8, time_warp_mag=0.1, channel_drop_prob=0.1,
                 is_train=True):
        self.X = X
        self.y = y
        self.max_jitter = max_jitter
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_prob = time_warp_prob
        self.time_warp_mag = time_warp_mag
        self.channel_drop_prob = channel_drop_prob
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone() # Clone to avoid in-place modification of dataset array
        y_val = self.y[idx]
        if self.is_train:
            # 1. Random Crop (Replaces padding-based temporal jitter)
            # The input sequence is EXTENDED_SEQ_LEN (e.g. 230). We need to crop it down to SEQ_LEN (e.g. 200).
            seq_len = 200
            if x.shape[-1] > seq_len:
                max_start_idx = x.shape[-1] - seq_len
                start_idx = np.random.randint(0, max_start_idx + 1) if self.max_jitter > 0 else max_start_idx // 2
                x = x[..., start_idx:start_idx + seq_len]
            elif x.shape[-1] == seq_len:
                pass
            else:
                raise ValueError(f"Input sequence length {x.shape[-1]} is shorter than target length {seq_len}")
            
            # 2. Time Warping (non-linear temporal deformation)
            # Simulates variable vehicle speed by warping the time axis
            # using a smooth random curve (4 control points, cubic interp)
            if self.time_warp_prob > 0 and np.random.rand() < self.time_warp_prob:
                T = x.shape[-1]
                # Generate smooth warping curve with 4 control points
                n_knots = 4
                knot_positions = np.linspace(0, T - 1, n_knots + 2)
                knot_offsets = np.random.uniform(-self.time_warp_mag * T, 
                                                  self.time_warp_mag * T, 
                                                  size=n_knots + 2)
                knot_offsets[0] = 0  # Anchor start
                knot_offsets[-1] = 0  # Anchor end
                
                # Interpolate warping offsets to all timesteps
                orig_indices = np.arange(T, dtype=np.float32)
                warped_indices = np.interp(orig_indices, knot_positions, 
                                           knot_positions + knot_offsets)
                # Clip to valid range
                warped_indices = np.clip(warped_indices, 0, T - 1)
                
                # Resample using linear interpolation
                warped_int = warped_indices.astype(np.int64)
                warped_frac = warped_indices - warped_int
                warped_int_next = np.minimum(warped_int + 1, T - 1)
                
                warped_frac_t = torch.from_numpy(warped_frac).float().unsqueeze(0)
                x = x[:, warped_int] * (1 - warped_frac_t) + x[:, warped_int_next] * warped_frac_t
            
            # 3. Gaussian Noise Injection
            if self.noise_std > 0 and np.random.rand() < 1.0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
            
            # 4. Magnitude Scaling (per-channel random scale)
            if self.scale_range is not None and np.random.rand() < 1.0:
                lo, hi = self.scale_range
                n_channels = x.shape[0]
                scale = torch.FloatTensor(n_channels, 1).uniform_(lo, hi)
                x = x * scale
            
            # 5. Channel Dropout (zero out a random channel)
            # Simulates sensor fault or device orientation change
            if self.channel_drop_prob > 0 and np.random.rand() < self.channel_drop_prob:
                n_channels = x.shape[0]
                drop_idx = np.random.randint(0, n_channels)
                x[drop_idx, :] = 0.0
                
        else:
            # During evaluation, strictly use center crop
            seq_len = 200
            if x.shape[-1] > seq_len:
                start_idx = (x.shape[-1] - seq_len) // 2
                x = x[..., start_idx:start_idx + seq_len]
                
        return x, y_val

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate raw CE loss WITHOUT weights to get correct pt mathematically
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=None)
        pt = torch.exp(-ce_loss)  # probability of correct prediction
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply class weights manually if provided
        if self.weight is not None:
            target_weights = self.weight[targets]
            focal_loss = focal_loss * target_weights

        if self.reduction == 'mean':
            if self.weight is not None:
                return focal_loss.sum() / target_weights.sum()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def main():
    parser = argparse.ArgumentParser(description="CNN Training")
    parser.add_argument("--channels", type=int, default=128, help="Number of base channels")
    parser.add_argument("--dropout", type=float, default=0.48195, help="Dropout rate")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    set_seed(42)
    X_path = os.path.join(DATA_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(DATA_DIR, "cnn_1d_y.npy")
    groups_path = os.path.join(DATA_DIR, "cnn_1d_groups.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(groups_path)):
        logger.error("Data tidak ditemukan. Jalankan build_cnn_data.py terlebih dahulu.")
        return
        
    X_all = np.load(X_path)
    y_raw_all = np.load(y_path)
    groups_all = np.load(groups_path)
    
    # Stratified split 70% Dev Set, 30% Holdout Test Set based on trip_id
    dev_groups_list, test_groups_list = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.7)
    
    dev_mask = np.isin(groups_all, dev_groups_list)
    test_mask = np.isin(groups_all, test_groups_list)
    
    # Dev data for K-fold CV and training the final model
    X = X_all[dev_mask]
    y_raw = y_raw_all[dev_mask]
    groups = groups_all[dev_mask]
    
    # Holdout Test data for final evaluation
    X_test_np = X_all[test_mask]
    y_test_raw = y_raw_all[test_mask]
    groups_test = groups_all[test_mask]
    
    logger.info(f"Split Summary (Trip-Based):")
    logger.info(f"  Dev Set (70%): {len(dev_groups_list)} trips, {len(X)} samples")
    logger.info(f"  Test Set (30%): {len(test_groups_list)} trips, {len(X_test_np)} samples")
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    y_test = le.transform(y_test_raw)  # Convert test labels to numeric
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []
    
    best_epochs = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"=== Fold {fold+1} ===")
        
        # ─── SMOTE Oversampling (Disabled for Experiment) ───
        X_train_np, y_train_np = X[train_idx], y[train_idx]
        
        # Calculate global mean and std from X_train_np (N, C, T) over N and T (axis=(0, 2))
        global_means = np.mean(X_train_np, axis=(0, 2), keepdims=True)
        global_stds = np.std(X_train_np, axis=(0, 2), keepdims=True)
        global_stds = np.where(global_stds < 1e-6, 1.0, global_stds)
        
        if fold == 0:
            scaler_params = {
                "means": global_means.flatten().tolist(),
                "stds": global_stds.flatten().tolist()
            }
            scaler_path = os.path.join(MODEL_DIR, "scaler_params.json")
            with open(scaler_path, "w") as f:
                json.dump(scaler_params, f, indent=4)
            logger.info(f"Saved fold scaler parameters to {scaler_path}")
            
        X_train_scaled = (X_train_np - global_means) / global_stds
        # Apply the SAME training means and stds to validation data
        X_val_np = X[val_idx]
        X_val_scaled = (X_val_np - global_means) / global_stds
        
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=len(classes)).float()
        
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)
        y_val_one_hot = torch.nn.functional.one_hot(y_val, num_classes=len(classes)).float()
        
        # ─── Stratified Batch Loader ───
        if args.no_augment:
            max_j = 0
            n_std = 0
            s_range = None
            drop_p = 0
            warp_p = 0
        else:
            max_j = 15
            n_std = 0.02
            s_range = (0.85, 1.15)
            drop_p = 0.1
            warp_p = 0.8
            
        train_dataset = DynamicJitterDataset(X_train, y_train, max_jitter=max_j,
                                             noise_std=n_std, scale_range=s_range,
                                             time_warp_prob=warp_p, channel_drop_prob=drop_p,
                                             is_train=True)
        val_dataset = DynamicJitterDataset(X_val, y_val, max_jitter=0,
                                           noise_std=0, scale_range=None, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Dampened Class Weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        model = InceptionTime1D(in_channels=X_train.shape[1], num_classes=len(classes),
                               num_blocks=3, channels=args.channels, bottleneck_channels=args.channels//4, dropout_rate=args.dropout).to(device)
        criterion = MultiClassFocalLoss(weight=None, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        best_val_loss = float('inf')
        best_val_prauc = -1.0
        best_model_state = None
        best_epoch_idx = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_loader.dataset)
            scheduler.step()
            
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
                    
                    # Gunakan Softmax untuk Multi-class
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1) # Default evaluation argmax
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_probas.extend(probs.cpu().numpy())
                    val_trues.extend(batch_y.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            
            logger.info(f"  [Epoch {epoch+1:02d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Hitung PR-AUC untuk kelas minoritas pada data validasi epoch ini (threshold-independent)
            val_probas_np = np.array(val_probas)
            val_trues_np = np.array(val_trues)
            
            # PR-AUC Pothole
            y_val_pothole = (val_trues_np == p_idx).astype(int)
            y_proba_pothole = val_probas_np[:, p_idx]
            prec_p, rec_p, _ = precision_recall_curve(y_val_pothole, y_proba_pothole)
            pr_auc_pothole = auc(rec_p, prec_p)
            
            # PR-AUC Speed Bump
            if sb_idx != -1:
                y_val_sb = (val_trues_np == sb_idx).astype(int)
                y_proba_sb = val_probas_np[:, sb_idx]
                prec_sb, rec_sb, _ = precision_recall_curve(y_val_sb, y_proba_sb)
                pr_auc_sb = auc(rec_sb, prec_sb)
                val_minority_prauc = (pr_auc_pothole + pr_auc_sb) / 2.0
            else:
                val_minority_prauc = pr_auc_pothole
            
            # Kriteria penyimpanan: Prioritaskan PR-AUC kelas minoritas tertinggi,
            # jika sama, gunakan loss terendah sebagai tie-breaker.
            is_better = False
            if val_minority_prauc > best_val_prauc:
                is_better = True
            elif np.isclose(val_minority_prauc, best_val_prauc) and val_loss < best_val_loss:
                is_better = True
                
            if is_better:
                best_val_prauc = val_minority_prauc
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch_idx = epoch + 1
                
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        best_epochs.append(best_epoch_idx)
        
        # Calculate final fold metrics
        model.eval()
        all_probas = []
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.nn.functional.softmax(outputs, dim=1)
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
        logger.info(f"Fold {fold+1} Best Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch_idx} | Val Pothole F1: {f1_val:.4f} | Val PR-AUC: {pr_auc_val:.4f}")
        
        oof_y_true.extend(all_trues)
        oof_y_pred.extend(all_preds)
        oof_y_proba.extend(all_probas)
        
    avg_f1 = np.mean([m[0] for m in fold_metrics])
    avg_prauc = np.mean([m[1] for m in fold_metrics])
    optimal_epochs = int(np.round(np.mean(best_epochs)))
    print("-" * 60)
    logger.info(f"Average Pothole F1: {avg_f1:.4f} | Average PR-AUC: {avg_prauc:.4f}")
    logger.info(f"Best epochs per fold: {best_epochs} | Optimal Epoch Average: {optimal_epochs}")
    
    # --- Cetak Classification Report ---
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)
    
    from sklearn.metrics import classification_report
    
    print("\n" + "=" * 60)
    print("      OOF REPORT      ")
    print("=" * 60)
    print(classification_report(oof_y_true, oof_y_pred, target_names=classes))
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(oof_y_true, oof_y_pred)
    
    print("-" * 60)
    print("        CONFUSION MATRIX (Out-of-Fold)")
    print("-" * 60)
    header = f"{'':>12}" + "".join([f"{cls:>12}" for cls in classes])
    print(header)
    print("-" * len(header))
    for i, row_label in enumerate(classes):
        row = f"{row_label:>12}" + "".join([f"{cm[i, j]:>12}" for j in range(len(classes))])
        print(row)
    print("=" * 60)
    
    # --- Simpan Confusion Matrix sebagai gambar ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Out-of-Fold)')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells with values and percentages
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row_total = cm[i].sum()
            pct = cm[i, j] / row_total * 100 if row_total > 0 else 0
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight='bold',
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "cnn_1d_confusion_matrix_oof.png")
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Confusion matrix disimpan di {cm_path}")
    
    # --- Simpan OOF predictions untuk analisis ulang ---
    np.save(os.path.join(MODEL_DIR, "cnn_1d_oof_y_true.npy"), oof_y_true)
    np.save(os.path.join(MODEL_DIR, "cnn_1d_oof_y_pred.npy"), oof_y_pred)
    np.save(os.path.join(MODEL_DIR, "cnn_1d_oof_y_proba.npy"), oof_y_proba)
    logger.info("OOF predictions disimpan untuk analisis ulang.")
    
    # --- Train Final Model ---
    X_full, y_full = X, y
    
    # ─── Fit and Save Final Scaler ───
    final_means = np.mean(X_full, axis=(0, 2), keepdims=True)
    final_stds = np.std(X_full, axis=(0, 2), keepdims=True)
    final_stds = np.where(final_stds < 1e-6, 1.0, final_stds)
    
    final_scaler_params = {
        "means": final_means.flatten().tolist(),
        "stds": final_stds.flatten().tolist()
    }
    scaler_path = os.path.join(MODEL_DIR, "cnn_1d_scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(final_scaler_params, f, indent=4)
    logger.info(f"Saved final global scaler parameters to {scaler_path}")
    
    X_full_scaled = (X_full - final_means) / final_stds
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full, dtype=torch.long)
    
    logger.info(f"Melatih final model dengan seluruh dataset sebanyak {optimal_epochs} epoch...")
    
    full_dataset = DynamicJitterDataset(X_full_tensor, y_full_tensor, max_jitter=max_j,
                                         noise_std=n_std, scale_range=s_range, 
                                         time_warp_prob=warp_p, channel_drop_prob=drop_p,
                                         is_train=True)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_weights_full = compute_class_weight('balanced', classes=np.unique(y_full), y=y_full)
    # class_weights_full = np.sqrt(class_weights_full)  # Dihapus agar konsisten dengan cross-validation
    class_weights_full = class_weights_full / class_weights_full.sum() * len(class_weights_full)
    class_weights_full = torch.tensor(class_weights_full, dtype=torch.float32).to(device)
    
    final_model = InceptionTime1D(in_channels=X_full.shape[1], num_classes=len(classes),
                                 num_blocks=3, channels=args.channels, bottleneck_channels=args.channels//4, dropout_rate=args.dropout).to(device)
    criterion_full = MultiClassFocalLoss(weight=None, gamma=2.0)
    optimizer_full = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=EPOCHS, eta_min=1e-6)
    
    for epoch in range(optimal_epochs):
        final_model.train()
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_full.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion_full(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer_full.step()
        scheduler_full.step()
            
    # Save Model
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, "cnn_1d_model.pth"))
    
    # Save Label Encoder Classes
    np.save(os.path.join(MODEL_DIR, "cnn_1d_classes.npy"), classes)
    
    logger.info("Model final PyTorch disimpan di " + os.path.join(MODEL_DIR, "cnn_1d_model.pth"))
    
    # --- Evaluate Final Model on Holdout Test Set (30%) ---
    logger.info("Mengevaluasi model final pada Holdout Test Set (30%)...")
    final_model.eval()
    
    # Standardize Test set using global scaling
    X_test_scaled = (X_test_np - final_means) / final_stds
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = DynamicJitterDataset(X_test_tensor, y_test_tensor, max_jitter=0,
                                        noise_std=0, scale_range=None, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_probas = []
    test_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = final_model(batch_x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            test_probas.extend(probs.cpu().numpy())
            test_trues.extend(batch_y.numpy())
            
    test_probas = np.array(test_probas)
    test_trues = np.array(test_trues)
    
    # --- Report Default Argmax on Holdout ---
    test_preds = np.argmax(test_probas, axis=1)
    print("\n" + "=" * 60)
    print("     HOLDOUT TEST     ")
    print("=" * 60)
    print(classification_report(test_trues, test_preds, target_names=classes))
    print("=" * 60 + "\n")
    
    # Save test confusion matrix
    cm_test = confusion_matrix(test_trues, test_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_test, interpolation='nearest', cmap='Oranges')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm_test.shape[1]),
           yticks=np.arange(cm_test.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Holdout Test Set)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_test.max() / 2.0
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            row_total = cm_test[i].sum()
            pct = cm_test[i, j] / row_total * 100 if row_total > 0 else 0
            ax.text(j, i, f"{cm_test[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight='bold',
                    color="white" if cm_test[i, j] > thresh else "black")
    fig.tight_layout()
    cm_test_path = os.path.join(REPORT_DIR, "cnn_1d_confusion_matrix_holdout.png")
    fig.savefig(cm_test_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Holdout Test confusion matrix disimpan di {cm_test_path}")

    # ─── ONNX EXPORT ───
    logger.info("ONNX export is now handled exclusively by export_onnx.py to ensure MobileInferenceWrapper is applied.")
    logger.info("Please run `python src/cnn_model/export_onnx.py` manually after training.")

if __name__ == "__main__":
    main()
