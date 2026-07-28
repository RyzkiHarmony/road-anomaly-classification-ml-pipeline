import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Path untuk memuat probabilitas dari model Baseline dan Optuna
baseline_true_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\baseline_holdout_y_true.npy"
baseline_proba_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\baseline_holdout_y_proba.npy"
optuna_true_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\optuna_holdout_y_true.npy"
optuna_proba_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\optuna_holdout_y_proba.npy"

output_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\reports\cnn_1d\cnn_1d_multiclass_pr_curve.png"

classes = ['Non-Event', 'Pothole', 'Speed Bump']
colors = ['green', 'darkorange', 'purple']

def plot_multiclass():
    if not (os.path.exists(baseline_true_path) and os.path.exists(optuna_true_path)):
        print("File prediksi belum lengkap.")
        return

    # Load arrays
    y_true_base = np.load(baseline_true_path)
    y_proba_base = np.load(baseline_proba_path)
    y_true_opt = np.load(optuna_true_path)
    y_proba_opt = np.load(optuna_proba_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # --- PLOT 1: BASELINE ---
    for i, cls_name in enumerate(classes):
        y_test_bin = (y_true_base == i).astype(int)
        prob = y_proba_base[:, i]
        prec, rec, _ = precision_recall_curve(y_test_bin, prob)
        pr_auc = auc(rec, prec)
        axes[0].plot(rec, prec, color=colors[i], lw=2.5, label=f'{cls_name} (AUC = {pr_auc:.4f})')
        
    axes[0].set_title('Baseline 1D-CNN', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].legend(loc='lower left', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- PLOT 2: OPTUNA TUNED ---
    for i, cls_name in enumerate(classes):
        y_test_bin = (y_true_opt == i).astype(int)
        prob = y_proba_opt[:, i]
        prec, rec, _ = precision_recall_curve(y_test_bin, prob)
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, color=colors[i], lw=2.5, label=f'{cls_name} (AUC = {pr_auc:.4f})')
        
    axes[1].set_title('Optuna Tuned 1D-CNN', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc='lower left', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('Multiclass Precision-Recall Curve (Holdout Test)', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    print(f"Selesai! Multiclass PR-Curve telah disimpan ke {output_path}")

if __name__ == "__main__":
    plot_multiclass()
