import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Tambahkan path ke utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import OUT_FOLDER

def focal_loss(p, gamma):
    # p = predicted probability for the true class
    # batasi p agar tidak log(0)
    p = np.clip(p, 1e-8, 1. - 1e-8)
    return - (1 - p) ** gamma * np.log(p)

def main():
    p = np.linspace(0.01, 1.0, 100)
    
    # Cross Entropy (gamma = 0)
    loss_ce = focal_loss(p, 0)
    # Focal Loss (gamma = 1)
    loss_fl_1 = focal_loss(p, 1)
    # Focal Loss Baseline (gamma = 2.0)
    loss_fl_2 = focal_loss(p, 2.0)
    # Focal Loss Optuna Tuned (gamma = 2.09)
    loss_fl_opt = focal_loss(p, 2.09)

    plt.figure(figsize=(10, 6))
    plt.plot(p, loss_ce, 'k--', linewidth=2, label='Cross-Entropy ($\gamma=0$)')
    plt.plot(p, loss_fl_1, 'b-', linewidth=2, label='Focal Loss ($\gamma=1$)')
    plt.plot(p, loss_fl_2, 'g-', linewidth=2, label='Focal Loss Baseline ($\gamma=2.0$)')
    plt.plot(p, loss_fl_opt, 'r-', linewidth=2, label='Focal Loss Optuna ($\gamma=2.09$)')

    plt.title("Komparasi Fungsi Kerugian: Cross-Entropy vs Focal Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Probabilitas Prediksi Kelas Benar ($p_t$)", fontsize=12)
    plt.ylabel("Nilai Penalti Kerugian (*Loss*)", fontsize=12)
    
    # Highlight area easy examples
    plt.axvspan(0.6, 1.0, color='gray', alpha=0.1, label='Sampel Mayoritas / Easy Examples (p > 0.6)')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    out_dir = os.path.join(OUT_FOLDER, "reports", "cnn_1d")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "focal_loss_curve.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Visualisasi Focal Loss berhasil disimpan di: {out_path}")

if __name__ == "__main__":
    main()
