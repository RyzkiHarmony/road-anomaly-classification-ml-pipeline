# plot_comparisons.py
# Tahap 6: Visualisasi Perbandingan Model (Akurasi, F1-Score, dan Latensi Komputasi).

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "reports")
COMP_CSV = os.path.join(REPORT_DIR, "comparison_report.csv")
BENCH_TXT = os.path.join(REPORT_DIR, "inference_benchmark_report.txt")
OUTPUT_PLOT = os.path.join(REPORT_DIR, "model_comparison_barchart.png")


def main():
    print("Membuat grafik komparasi performa dan efisiensi komputasi model...")

    # Data performa default dari pengujian holdout test
    classes = ["Non-Event", "Pothole", "Speed Bump"]
    xgb_f1 = [0.9732, 0.6040, 0.6562]
    cnn_f1 = [0.9751, 0.7800, 0.8100]

    xgb_macro_f1 = np.mean(xgb_f1)
    cnn_macro_f1 = np.mean(cnn_f1)

    xgb_latency_ms = 0.0172
    cnn_latency_ms = 0.2040

    if os.path.exists(COMP_CSV):
        try:
            df = pd.read_csv(COMP_CSV)
            if "Class" in df.columns:
                classes = df["Class"].tolist()
                xgb_f1 = df["XGBoost F1-score"].tolist()
                cnn_f1 = df["1D-CNN F1-score"].tolist()
                xgb_macro_f1 = float(np.mean(xgb_f1))
                cnn_macro_f1 = float(np.mean(cnn_f1))
        except Exception as e:
            print(f"Peringatan: Gagal membaca {COMP_CSV}: {e}")

    # Plot Visualisasi 2 Subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: Perbandingan F1-Score Per Kelas & Macro F1 ---
    x = np.arange(len(classes) + 1)
    width = 0.35

    all_labels = classes + ["Macro F1"]
    all_xgb = xgb_f1 + [xgb_macro_f1]
    all_cnn = cnn_f1 + [cnn_macro_f1]

    bars1 = ax1.bar(x - width/2, all_xgb, width, label='XGBoost', color='#2b5c8f', edgecolor='black')
    bars2 = ax1.bar(x + width/2, all_cnn, width, label='1D-CNN', color='#e66101', edgecolor='black')

    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Komparasi F1-Score Per Kelas & Macro F1', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b34700')

    # --- Subplot 2: Komparasi Latensi Inferensi CPU (ms/window) ---
    models = ['XGBoost ONNX', '1D-CNN ONNX']
    latencies = [xgb_latency_ms, cnn_latency_ms]
    colors = ['#2b5c8f', '#e66101']

    bars_lat = ax2.bar(models, latencies, width=0.45, color=colors, edgecolor='black')
    ax2.axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='Batas Toleransi Real-Time (10 ms)')
    ax2.set_ylabel('Latensi Inferensi Rata-Rata (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Efisiensi Komputasi Inferensi Edge CPU', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(latencies) * 2.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True)

    for bar in bars_lat:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(latencies)*0.05),
                 f"{yval:.4f} ms\n({1000.0/yval:.0f} FPS)", ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Diagram batang komparasi berhasil disimpan di: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
