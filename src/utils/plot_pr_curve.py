import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

def plot_pr_curve(model_dir, report_dir):
    # Construct file paths
    y_true_path = os.path.join(model_dir, "cnn_1d_oof_y_true.npy")
    y_proba_path = os.path.join(model_dir, "cnn_1d_oof_y_proba.npy")
    classes_path = os.path.join(model_dir, "cnn_1d_classes.npy")

    # Check if files exist
    if not all([os.path.exists(p) for p in [y_true_path, y_proba_path, classes_path]]):
        print(f"Error: Missing satu atau lebih file .npy di folder: {model_dir}")
        print("Pastikan model sudah ditraining dan file oof_y_true.npy, oof_y_proba.npy, serta classes.npy telah ter-generate.")
        return

    # Load data
    y_true = np.load(y_true_path)
    y_proba = np.load(y_proba_path)
    classes = np.load(classes_path)
    n_classes = len(classes)

    # Binarize the true labels untuk perhitungan per-kelas
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Setup the plot
    plt.figure(figsize=(9, 7))
    
    # Warna yang digunakan untuk membedakan garis tiap kelas
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Hitung dan Plot PR Curve untuk masing-masing kelas
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2.5,
                 label=f'{classes[i]} (PR-AUC = {pr_auc:.4f})')

    # Format plot
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Out-of-Fold Validation)', fontsize=14, pad=15)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    # Buat direktori report jika belum ada
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "cnn_1d_pr_curve.png")
    
    # Simpan dan Tampilkan
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gambar PR Curve berhasil disimpan di:\n -> {output_path}")
    
    # Show window only if not in headless environment (we will just close it in headless)
    try:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Precision-Recall Curve dari data OOF")
    parser.add_argument("--model_dir", type=str, 
                        default=r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d", 
                        help="Folder tempat cnn_1d_oof_y_true.npy disimpan")
    parser.add_argument("--report_dir", type=str, 
                        default=r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\reports\cnn_1d", 
                        help="Folder tempat gambar PR Curve akan disimpan")
    
    args = parser.parse_args()
    
    plot_pr_curve(args.model_dir, args.report_dir)
