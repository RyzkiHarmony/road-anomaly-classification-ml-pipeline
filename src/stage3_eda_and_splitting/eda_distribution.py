# eda_distribution.py
# Tahap 3: Analisis Eksplorasi Data (EDA) - Distribusi Kelas Target & Deteksi Imbalance.

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))

from config import OUT_FOLDER, XGB_OUT_DIR, get_logger

logger = get_logger(__name__)


def main():
    logger.info("Memulai Analisis Distribusi Kelas Target (EDA)...")

    # Muat dataset terlabeli
    xgb_data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    gt_path = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
    events_path = os.path.join(OUT_FOLDER, "candidates_events.csv")
    shared_bg_path = os.path.join(OUT_FOLDER, "shared_background.csv")

    df = None
    if os.path.exists(xgb_data_path):
        df = pd.read_csv(xgb_data_path)
    elif os.path.exists(gt_path) and os.path.exists(events_path):
        df_gt = pd.read_csv(gt_path)
        df_events = pd.read_csv(events_path)
        df = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
        if os.path.exists(shared_bg_path):
            df_bg = pd.read_csv(shared_bg_path)
            df = pd.concat([df, df_bg], ignore_index=True)
    else:
        logger.error("Dataset ground truth tidak ditemukan.")
        return

    df = df.dropna(subset=['label'])
    counts = df['label'].value_counts()
    total = len(df)

    logger.info(f"Total Sampel: {total}")
    for cls_name, count in counts.items():
        pct = (count / total) * 100
        logger.info(f"  - {cls_name:<12}: {count:>5} sampel ({pct:.2f}%)")

    # Hitung rasio imbalance terhadap kelas minoritas
    non_event_count = counts.get("Non-Event", 0)
    pothole_count = counts.get("Pothole", 1)
    sb_count = counts.get("Speed Bump", 1)

    ratio_pothole = non_event_count / pothole_count if pothole_count > 0 else 0
    ratio_sb = non_event_count / sb_count if sb_count > 0 else 0
    logger.info(f"Rasio Imbalance Non-Event : Pothole     = {ratio_pothole:.2f} : 1")
    logger.info(f"Rasio Imbalance Non-Event : Speed Bump = {ratio_sb:.2f} : 1")

    # Visualisasi Bar Chart Distribusi Kelas
    report_dir = os.path.join(_PROJECT_ROOT, "evaluation", "reports")
    os.makedirs(report_dir, exist_ok=True)
    plot_path = os.path.join(report_dir, "eda_class_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(counts.index)
    values = list(counts.values)
    colors = ['#2b5c8f', '#d95f02', '#7570b3'][:len(classes)]

    bars = ax.bar(classes, values, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Jumlah Sampel", fontsize=12, fontweight='bold')
    ax.set_title("Distribusi Kelas Target & Analisis Ketidakseimbangan Data", fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Tambahkan anotasi teks di atas setiap batang
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values)*0.02),
                f"{int(yval)}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot distribusi kelas berhasil disimpan di: {plot_path}")


if __name__ == "__main__":
    main()
