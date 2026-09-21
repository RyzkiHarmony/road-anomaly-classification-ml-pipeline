import argparse
import json
import os
import random

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Non-interactive backend
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "stage3_eda_and_splitting"))
sys.path.append(os.path.dirname(__file__))

from config import CNN_OUT_DIR, get_logger
from data_splitting import get_stratified_group_split
from model import InceptionTime1D

logger = get_logger(__name__)

DATA_DIR = CNN_OUT_DIR
MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d")
REPORT_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "cnn_1d")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

#baseline
EPOCHS = 50
LR = 0.001
DROPOUT = 0.2
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0001
CHANNELS = 32
GAMMA = 2.0


sys.path.append(os.path.dirname(__file__))
from training_utils import DynamicJitterDataset, MultiClassFocalLoss, set_seed

def main():
    parser = argparse.ArgumentParser(description="CNN Training")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    # ─── Load Optuna Best Params ───
    best_params_path = os.path.join(os.path.dirname(__file__), "best_optuna_params.json")

    # Defaults
    lr_val = LR
    batch_size_val = BATCH_SIZE
    weight_decay_val = WEIGHT_DECAY
    gamma_val = GAMMA
    channels_val = CHANNELS
    dropout_val = DROPOUT

    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Loaded Optuna best parameters from {best_params_path}")
        lr_val = best_params.get("lr", lr_val)
        batch_size_val = best_params.get("batch_size", batch_size_val)
        channels_val = best_params.get("channels", channels_val)
        dropout_val = best_params.get("dropout", dropout_val)
        weight_decay_val = best_params.get("weight_decay", weight_decay_val)
        gamma_val = best_params.get("gamma", gamma_val)

        logger.info(f"Using Params -> LR: {lr_val:.6f}, Batch: {batch_size_val}, Channels: {channels_val}, Dropout: {dropout_val:.4f}, WD: {weight_decay_val:.6f}, Gamma: {gamma_val:.2f}")
    else:
        logger.warning(f"No best_params.json found at {best_params_path}. Using default parameters.")

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

    # Stratified split 80% Dev Set, 20% Holdout Test Set based on trip_id
    dev_groups_list, test_groups_list = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.8)

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

    logger.info("Split Summary (Trip-Based):")
    logger.info(f"  Dev Set (80%): {len(dev_groups_list)} trips, {len(X)} samples")
    logger.info(f"  Test Set (20%): {len(test_groups_list)} trips, {len(X_test_np)} samples")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    y_test = le.transform(y_test_raw)  # Convert test labels to numeric
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

    fold_metrics = []

    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"=== Fold {fold+1} ===")

        X_train_np, y_train_np = X[train_idx], y[train_idx]

        # Calculate global mean and std from X_train_np (N, C, T) over N and T (axis=(0, 2))
        global_means = np.mean(X_train_np, axis=(0, 2), keepdims=True)
        global_stds = np.std(X_train_np, axis=(0, 2), keepdims=True)
        global_stds = np.where(global_stds < 1e-6, 1.0, global_stds)

        X_train_scaled = (X_train_np - global_means) / global_stds
        # Apply the SAME training means and stds to validation data (no leakage)
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

        model = InceptionTime1D(in_channels=X_train.shape[1], num_classes=len(classes),
                               num_blocks=3, channels=channels_val, bottleneck_channels=channels_val//4, dropout_rate=dropout_val).to(device)
        criterion = MultiClassFocalLoss(weight=None, gamma=gamma_val)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=weight_decay_val)
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

            val_probas_np = np.array(val_probas)
            val_trues_np = np.array(val_trues)
            val_preds_np = np.array(val_preds)
            val_macro_f1 = f1_score(val_trues_np, val_preds_np, average='macro', zero_division=0)

            logger.info(f"  [Epoch {epoch+1:02d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f}")

            # Hitung PR-AUC untuk kelas minoritas pada data validasi epoch ini (threshold-independent)

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

    # Load best params
    best_params_path = os.path.join(os.path.dirname(__file__), "best_optuna_params.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            bp = json.load(f)
            lr_val = bp.get('lr', LR)
            batch_size_val = bp.get('batch_size', BATCH_SIZE)
            dropout_val = bp.get('dropout', DROPOUT)
            channels_val = bp.get('channels', CHANNELS)
            weight_decay_val = bp.get('weight_decay', WEIGHT_DECAY)
            gamma_val = bp.get('gamma', GAMMA)
    else:
        lr_val, batch_size_val, dropout_val, channels_val, weight_decay_val, gamma_val = LR, BATCH_SIZE, DROPOUT, CHANNELS, WEIGHT_DECAY, GAMMA

    full_loader = DataLoader(full_dataset, batch_size=batch_size_val, shuffle=True)

    final_model = InceptionTime1D(in_channels=X_full.shape[1], num_classes=len(classes),
                                 num_blocks=3, channels=channels_val, bottleneck_channels=channels_val//4, dropout_rate=dropout_val).to(device)
    final_criterion = MultiClassFocalLoss(weight=None, gamma=gamma_val)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=lr_val, weight_decay=weight_decay_val)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=optimal_epochs, eta_min=1e-6)

    for epoch in range(optimal_epochs):
        final_model.train()
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = final_criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            final_optimizer.step()
        final_scheduler.step()

    # Save Model
    torch.save(final_model.state_dict(), os.path.join(MODEL_DIR, "cnn_1d_model.pth"))

    # Save Label Encoder Classes
    np.save(os.path.join(MODEL_DIR, "cnn_1d_classes.npy"), classes)

    logger.info("Model final PyTorch disimpan di " + os.path.join(MODEL_DIR, "cnn_1d_model.pth"))

    # --- Evaluate Final Model on Holdout Test Set (20%) ---
    logger.info("Mengevaluasi model final pada Holdout Test Set (20%)...")
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

    # Calculate PR-AUC for Holdout
    y_test_pothole = (test_trues == p_idx).astype(int)
    y_proba_pothole_test = test_probas[:, p_idx]
    prec_test, rec_test, _ = precision_recall_curve(y_test_pothole, y_proba_pothole_test)
    pr_auc_test = auc(rec_test, prec_test)
    logger.info(f"Holdout Test Pothole PR-AUC: {pr_auc_test:.4f}")

    # Save Holdout predictions for PR-Curve analysis
    np.save(os.path.join(MODEL_DIR, "cnn_1d_holdout_y_true.npy"), test_trues)
    np.save(os.path.join(MODEL_DIR, "cnn_1d_holdout_y_proba.npy"), test_probas)
    logger.info("Holdout Test predictions saved for PR-Curve analysis.")

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
    logger.info("ONNX export is now handled exclusively by 08_export_onnx.py to ensure MobileInferenceWrapper is applied.")
    logger.info("Please run `python src/stage6_reports_deployment/08_export_onnx.py` manually after training.")

if __name__ == "__main__":
    main()
