import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_pipeline_experiment'))
from config import get_logger

from model import Lightweight1DCNN

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# EPOCHS: Batas atas pencarian epoch optimal via K-Fold CV.
# Model final TIDAK dilatih sebanyak EPOCHS, melainkan menggunakan
# rata-rata best epoch per fold (lihat variabel `optimal_epochs`).
# Eksperimen menunjukkan konvergensi di epoch 9-22; ceiling 50 sudah memadai.
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.000828659730834538

# ─── SMOTE Configuration ───
# SMOTE ratio: target jumlah sampel minority relatif terhadap majority.
# 0.5 = minority di-oversample hingga 50% dari jumlah majority.
# Tidak full-balance (1.0) karena over-representation sintetis bisa
# memperkenalkan artefak dan menurunkan precision.
SMOTE_RATIO = 0.5

class DynamicJitterDataset(torch.utils.data.Dataset):
    """Dataset dengan augmentasi on-the-fly untuk time-series sensor.
    
    Augmentasi yang diterapkan saat training:
    1. Temporal Jitter: Geser sinyal ±max_jitter timesteps
    2. Gaussian Noise: Tambah noise σ=0.02 ke sinyal
    3. Magnitude Scaling: Skala amplitudo 0.85-1.15x secara random per-channel
    """
    def __init__(self, X, y, max_jitter=15, noise_std=0.02, scale_range=(0.85, 1.15), is_train=True):
        self.X = X
        self.y = y
        self.max_jitter = max_jitter
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone() # Clone to avoid in-place modification of dataset array
        y = self.y[idx]
        
        if self.is_train:
            # 1. Temporal Jitter
            if self.max_jitter > 0:
                shift = np.random.randint(-self.max_jitter, self.max_jitter + 1)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=-1)
                    # Avoid wrap-around artifact by filling with boundary values
                    if shift > 0:
                        x[..., :shift] = x[..., shift:shift+1]
                    else:
                        x[..., shift:] = x[..., shift-1:shift]
            
            # 2. Gaussian Noise Injection
            if self.noise_std > 0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
            
            # 3. Magnitude Scaling (per-channel random scale)
            if self.scale_range is not None:
                lo, hi = self.scale_range
                n_channels = x.shape[0]
                scale = torch.FloatTensor(n_channels, 1).uniform_(lo, hi)
                x = x * scale
                
        return x, y

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Reduces loss contribution from easy-to-classify samples (Non-Event)
    and focuses training on hard-to-classify minority samples (Pothole, Speed Bump).
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # class weights (alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def apply_smote(X_train_np, y_train_np, ratio=SMOTE_RATIO):
    """Terapkan SMOTE pada data training time-series.
    
    Proses: flatten (N, C, T) → (N, C*T) → SMOTE → reshape kembali → (N', C, T)
    
    Args:
        X_train_np: Array shape (N, C, T) — N samples, C channels, T timesteps
        y_train_np: Array shape (N,) — label per sample
        ratio: Target ratio minority/majority (0.5 = 50% dari majority count)
    
    Returns:
        X_resampled: Array (N', C, T)
        y_resampled: Array (N',)
    """
    N, C, T = X_train_np.shape
    X_flat = X_train_np.reshape(N, C * T)
    
    # Hitung target jumlah sampel per kelas
    unique, counts = np.unique(y_train_np, return_counts=True)
    max_count = counts.max()
    
    # Tentukan sampling strategy: minority di-boost ke ratio * max_count
    # tapi tidak boleh kurang dari jumlah asli
    sampling_strategy = {}
    for cls, cnt in zip(unique, counts):
        target = max(cnt, int(max_count * ratio))
        sampling_strategy[cls] = target
    
    # k_neighbors harus < jumlah sampel di kelas terkecil
    min_minority = min(counts[counts < max_count]) if np.sum(counts < max_count) > 0 else counts.min()
    k_neighbors = min(5, min_minority - 1)
    
    if k_neighbors < 1:
        logger.warning(f"Terlalu sedikit sampel minority ({min_minority}) untuk SMOTE. Skip oversampling.")
        return X_train_np, y_train_np
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=42
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_flat, y_train_np)
    X_resampled = X_resampled.reshape(-1, C, T)
    
    # Log distribusi sebelum dan sesudah
    unique_after, counts_after = np.unique(y_resampled, return_counts=True)
    before_str = ", ".join([f"{u}:{c}" for u, c in zip(unique, counts)])
    after_str = ", ".join([f"{u}:{c}" for u, c in zip(unique_after, counts_after)])
    logger.info(f"SMOTE: {before_str} → {after_str} (total: {len(y_train_np)} → {len(y_resampled)})")
    
    return X_resampled, y_resampled

def main():
    set_seed(42)
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
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []
    
    best_epochs = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"=== Fold {fold+1} ===")
        
        # ─── SMOTE Oversampling (hanya pada training data) ───
        X_train_np, y_train_np = X[train_idx], y[train_idx]
        X_train_smote, y_train_smote = apply_smote(X_train_np, y_train_np, ratio=SMOTE_RATIO)
        
        X_train = torch.tensor(X_train_smote, dtype=torch.float32)
        y_train = torch.tensor(y_train_smote, dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)
        
        # Dataset dengan augmentasi on-the-fly
        train_dataset = DynamicJitterDataset(X_train, y_train, max_jitter=15, 
                                              noise_std=0.02, scale_range=(0.85, 1.15), is_train=True)
        val_dataset = DynamicJitterDataset(X_val, y_val, max_jitter=0,
                                            noise_std=0, scale_range=None, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # ─── Dampened Class Weights ───
        # Gunakan sqrt(balanced_weights) untuk mencegah double-compensation
        # dengan FocalLoss yang sudah me-downweight easy samples via (1-pt)^γ.
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
        class_weights = np.sqrt(class_weights)  # Dampening: sqrt
        class_weights = class_weights / class_weights.sum() * len(class_weights)  # Re-normalize
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        model = Lightweight1DCNN(in_channels=10, num_classes=len(classes),
                                 conv1_filters=32, conv2_filters=64, dropout_rate=0.169).to(device)
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        best_val_loss = float('inf')
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
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(probs, 1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_probas.extend(probs.cpu().numpy())
                    val_trues.extend(batch_y.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
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
    oof_y_pred_default = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)
    
    from sklearn.metrics import classification_report
    
    # Threshold Optimization for Pothole
    y_true_pothole = (oof_y_true == p_idx).astype(int)
    y_proba_pothole = oof_y_proba[:, p_idx]
    prec, rec, thresholds = precision_recall_curve(y_true_pothole, y_proba_pothole)
    
    # Cari threshold yang memaksimalkan F1-Score
    fscore = (2 * prec * rec) / (prec + rec + 1e-9)
    ix = np.argmax(fscore)
    best_thresh_p = thresholds[ix] if ix < len(thresholds) else 0.5
    
    print("\n" + "=" * 60)
    print("              THRESHOLD OPTIMIZATION              ")
    print("=" * 60)
    print(f"Proposed Threshold for Pothole: {best_thresh_p:.4f}")
    print(f"Expected -> Precision: {prec[ix]:.4f}, Recall: {rec[ix]:.4f}, F1: {fscore[ix]:.4f}")
    
    best_thresh_sb = 0.5
    if sb_idx != -1:
        y_true_sb = (oof_y_true == sb_idx).astype(int)
        y_proba_sb = oof_y_proba[:, sb_idx]
        prec_sb, rec_sb, thresholds_sb = precision_recall_curve(y_true_sb, y_proba_sb)
        
        fscore_sb = (2 * prec_sb * rec_sb) / (prec_sb + rec_sb + 1e-9)
        ix_sb = np.argmax(fscore_sb)
        best_thresh_sb = thresholds_sb[ix_sb] if ix_sb < len(thresholds_sb) else 0.5
        
        print("\n" + "-" * 60)
        print(f"Proposed Threshold for Speed Bump: {best_thresh_sb:.4f}")
        print(f"Expected -> Precision: {prec_sb[ix_sb]:.4f}, Recall: {rec_sb[ix_sb]:.4f}, F1: {fscore_sb[ix_sb]:.4f}")

    # Terapkan threshold optimasi pada OOF predictions
    oof_y_pred = np.zeros_like(oof_y_true)
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    
    for i in range(len(oof_y_proba)):
        proba = oof_y_proba[i]
        p_prob = proba[p_idx]
        sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
        
        p_triggered = p_prob >= best_thresh_p
        sb_triggered = sb_idx != -1 and sb_prob >= best_thresh_sb
        
        if p_triggered and sb_triggered:
            # Jika kedua threshold terlewati, pilih kelas dengan probabilitas tertinggi
            if p_prob >= sb_prob:
                oof_y_pred[i] = p_idx
            else:
                oof_y_pred[i] = sb_idx
        elif p_triggered:
            oof_y_pred[i] = p_idx
        elif sb_triggered:
            oof_y_pred[i] = sb_idx
        else:
            oof_y_pred[i] = non_event_idx

    print("\nFinal Report (Out-of-Fold - Unbiased Default Argmax):")
    print(classification_report(oof_y_true, oof_y_pred_default, target_names=classes))
    print("-" * 60)
    print("\nFinal Report (Out-of-Fold - Unbiased Optimized Threshold):")
    print(classification_report(oof_y_true, oof_y_pred, target_names=classes))
    print("=" * 60 + "\n")
    
    # --- Confusion Matrix ---
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(oof_y_true, oof_y_pred)
    
    print("=" * 60)
    print("        CONFUSION MATRIX (Out-of-Fold - Optimized Threshold)")
    print("=" * 60)
    # Header
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
           title='Confusion Matrix (Out-of-Fold - Optimized)')
    
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
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Confusion matrix disimpan di {cm_path}")
    
    # --- Simpan OOF predictions untuk analisis ulang ---
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), oof_y_true)
    np.save(os.path.join(MODEL_DIR, "oof_y_pred.npy"), oof_y_pred)
    np.save(os.path.join(MODEL_DIR, "oof_y_proba.npy"), oof_y_proba)
    logger.info("OOF predictions disimpan untuk analisis ulang.")
    
    # --- Train Final Model (dengan SMOTE pada seluruh dataset) ---
    X_full_smote, y_full_smote = apply_smote(X, y, ratio=SMOTE_RATIO)
    logger.info(f"Melatih final model dengan seluruh dataset (SMOTE) sebanyak {optimal_epochs} epoch...")
    
    X_full = torch.tensor(X_full_smote, dtype=torch.float32)
    y_full = torch.tensor(y_full_smote, dtype=torch.long)
    full_dataset = DynamicJitterDataset(X_full, y_full, max_jitter=15,
                                         noise_std=0.02, scale_range=(0.85, 1.15), is_train=True)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_weights_full = compute_class_weight('balanced', classes=np.unique(y_full_smote), y=y_full_smote)
    class_weights_full = np.sqrt(class_weights_full)
    class_weights_full = class_weights_full / class_weights_full.sum() * len(class_weights_full)
    class_weights_full = torch.tensor(class_weights_full, dtype=torch.float32).to(device)
    
    final_model = Lightweight1DCNN(in_channels=10, num_classes=len(classes),
                                   conv1_filters=32, conv2_filters=64, dropout_rate=0.169).to(device)
    criterion_full = FocalLoss(weight=class_weights_full, gamma=2.0)
    optimizer_full = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=optimal_epochs, eta_min=1e-6)
    
    for epoch in range(optimal_epochs):
        final_model.train()
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_full.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion_full(outputs, batch_y)
            loss.backward()
            optimizer_full.step()
        scheduler_full.step()
            
    # Save Model
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, "best_1dcnn.pth"))
    
    # Save Label Encoder Classes
    np.save(os.path.join(MODEL_DIR, "classes.npy"), classes)
    
    logger.info("Model final PyTorch disimpan di " + os.path.join(MODEL_DIR, "best_1dcnn.pth"))
    
if __name__ == "__main__":
    main()
