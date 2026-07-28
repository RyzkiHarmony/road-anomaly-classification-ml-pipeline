import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Path untuk memuat probabilitas dari model Baseline
baseline_true_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\baseline_holdout_y_true.npy"
baseline_proba_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\baseline_holdout_y_proba.npy"

# Path untuk memuat probabilitas dari model Optuna
optuna_true_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\optuna_holdout_y_true.npy"
optuna_proba_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\optuna_holdout_y_proba.npy"

output_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\reports\cnn_1d\cnn_1d_pr_curve_comparison.png"

def plot_pr_curve():
    if not (os.path.exists(baseline_true_path) and os.path.exists(optuna_true_path)):
        print("File prediksi belum lengkap. Pastikan Anda telah menjalankan train.py untuk Baseline dan Optuna serta me-rename filenya sesuai instruksi.")
        return

    # Load arrays
    y_true_base = np.load(baseline_true_path)
    y_proba_base = np.load(baseline_proba_path)
    y_true_opt = np.load(optuna_true_path)
    y_proba_opt = np.load(optuna_proba_path)
    
    # Asumsi Pothole = Indeks 1 (Berdasarkan ['Non-Event', 'Pothole', 'Speed Bump'])
    p_idx = 1
    
    # Ekstrak biner (1 untuk Pothole, 0 untuk lainnya)
    y_test_base_bin = (y_true_base == p_idx).astype(int)
    y_test_opt_bin = (y_true_opt == p_idx).astype(int)
    
    # Probabilitas kelas Pothole
    prob_base = y_proba_base[:, p_idx]
    prob_opt = y_proba_opt[:, p_idx]
    
    # Hitung Kurva Precision-Recall dan AUC
    prec_base, rec_base, _ = precision_recall_curve(y_test_base_bin, prob_base)
    auc_base = auc(rec_base, prec_base)
    
    prec_opt, rec_opt, _ = precision_recall_curve(y_test_opt_bin, prob_opt)
    auc_opt = auc(rec_opt, prec_opt)
    
    # Plotting
    plt.figure(figsize=(9, 7))
    plt.plot(rec_base, prec_base, color='royalblue', lw=2.5, linestyle='--',
             label=f'1D-CNN Baseline (PR-AUC = {auc_base:.4f})')
    plt.plot(rec_opt, prec_opt, color='darkorange', lw=2.5, linestyle='-',
             label=f'1D-CNN Optuna Tuned (PR-AUC = {auc_opt:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.title('Precision-Recall Curve (Holdout Test - Pothole Class)', fontsize=15, fontweight='bold')
    plt.xlabel('Recall (Seberapa banyak Pothole yang tertangkap?)', fontsize=12)
    plt.ylabel('Precision (Seberapa akurat tebakan Pothole?)', fontsize=12)
    plt.legend(loc='lower left', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Selesai! PR-Curve telah disimpan ke {output_path}")

if __name__ == "__main__":
    plot_pr_curve()
