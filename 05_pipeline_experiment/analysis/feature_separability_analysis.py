# feature_separability_analysis.py
# Analisis separabilitas fitur event-level untuk klasifikasi:
# Normal vs Pothole (Lubang) vs Speed Bump (Polisi Tidur)

import sys
import os
# Ensure parent directory is in path for modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_SCRIPT_DIR))

import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


from config import OUT_FOLDER, get_logger, BEST_FEATURES

logger = get_logger(__name__)

# Resolve absolute paths
OUT_DIR = os.path.abspath(os.path.join(OUT_FOLDER, "separability"))
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATES_PATH = os.path.abspath(os.path.join(OUT_FOLDER, "candidates_events.csv"))
GT_PATH         = os.path.abspath(os.path.join(OUT_FOLDER, "ground_truth_labels.csv"))

TARGET_CLASSES = ["Non-Event", "Pothole", "Speed Bump"]

# Fitur kandidat yang akan dianalisis sekarang menggunakan BEST_FEATURES dari config.py

# Alias label yang sering muncul
NON_EVENT_ALIASES = {
    "non-event", "non event", "nonevent",
    "normal", "jalan mulus", "mulus", "bukan lubang",
    "non-anomaly", "non anomaly", "none", "no anomaly", "bukan anomali",
    "maneuver", "rough road", "crack", "jalan rusak", "jembatan",
    "severe anomaly",
}
POTHOLE_ALIASES = {
    "pothole", "lubang", "hole"
}
SPEED_BUMP_ALIASES = {
    "speed bump", "polisi tidur", "bump", "speedbump", "speed_bump"
}


# =========================
# HELPER
# =========================
def normalize_label(label):
    if pd.isna(label):
        return None
    s = str(label).strip().lower()
    if s in NON_EVENT_ALIASES:
        return "Non-Event"
    if s in POTHOLE_ALIASES:
        return "Pothole"
    if s in SPEED_BUMP_ALIASES:
        return "Speed Bump"
    return None


def cohen_d(x, y):
    """Effect size Cohen's d."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)

    if pooled <= 1e-12:
        return 0.0

    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def load_and_merge():
    if not os.path.exists(CANDIDATES_PATH):
        raise FileNotFoundError(f"Tidak menemukan {CANDIDATES_PATH}")
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Tidak menemukan {GT_PATH}")

    cand = pd.read_csv(CANDIDATES_PATH)
    gt = pd.read_csv(GT_PATH)

    if "event_id" not in cand.columns:
        raise ValueError("Kolom event_id tidak ada di candidates_events.csv")
    if "event_id" not in gt.columns or "label" not in gt.columns:
        raise ValueError("ground_truth_labels.csv harus punya kolom event_id dan label")

    gt = gt.copy()
    gt["label_norm"] = gt["label"].apply(normalize_label)
    gt = gt.dropna(subset=["label_norm"])

    # Jika ada relabel, ambil yang terakhir
    if "labeled_at" in gt.columns:
        gt = gt.sort_values("labeled_at")

    gt = gt.drop_duplicates(subset=["event_id"], keep="last")

    merged = cand.merge(
        gt[["event_id", "label_norm"]],
        on="event_id",
        how="inner"
    ).rename(columns={"label_norm": "class"})

    # Hanya kelas target
    merged = merged[merged["class"].isin(TARGET_CLASSES)].copy()

    if merged.empty:
        raise ValueError("Tidak ada data yang cocok setelah merge dan normalisasi label.")

    # Pastikan feature numerik
    for c in BEST_FEATURES:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged


def choose_available_features(df):
    feats = [c for c in BEST_FEATURES if c in df.columns]
    if not feats:
        raise ValueError("Tidak ada fitur yang tersedia dari BEST_FEATURES di dataset.")
    return feats


def compute_stats(df, features):
    rows = []
    pairwise_rows = []

    classes = [c for c in TARGET_CLASSES if c in df["class"].unique()]
    group_data = {cls: df[df["class"] == cls] for cls in classes}

    for feat in features:
        series_by_class = []
        valid_classes = []

        for cls in classes:
            vals = group_data[cls][feat].dropna()
            if len(vals) > 0:
                series_by_class.append(vals.values)
                valid_classes.append(cls)

        if len(series_by_class) >= 2:
            try:
                kw_stat, kw_p = kruskal(*series_by_class)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
        else:
            kw_stat, kw_p = np.nan, np.nan

        # Deskriptif per kelas
        row = {
            "feature": feat,
            "kruskal_stat": kw_stat,
            "kruskal_p": kw_p,
            "n_classes_present": len(valid_classes),
        }

        # Statistik kelas
        for cls in classes:
            vals = group_data[cls][feat].dropna()
            row[f"{cls}_n"] = len(vals)
            row[f"{cls}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{cls}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{cls}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{cls}_q25"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"{cls}_q75"] = float(vals.quantile(0.75)) if len(vals) else np.nan

        # Pairwise effect size
        pairwise_ds = []
        for a, b in itertools.combinations(classes, 2):
            da = group_data[a][feat].dropna()
            db = group_data[b][feat].dropna()
            d = cohen_d(da, db)
            pairwise_rows.append({
                "feature": feat,
                "class_a": a,
                "class_b": b,
                "cohen_d": d,
                "abs_cohen_d": abs(d) if pd.notna(d) else np.nan
            })
            if pd.notna(d):
                pairwise_ds.append(abs(d))

        row["max_abs_cohen_d"] = max(pairwise_ds) if pairwise_ds else np.nan
        row["mean_abs_cohen_d"] = float(np.mean(pairwise_ds)) if pairwise_ds else np.nan

        rows.append(row)

    stats_df = pd.DataFrame(rows).sort_values(
        by=["kruskal_p", "max_abs_cohen_d"],
        ascending=[True, False]
    )
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(
        by=["feature", "abs_cohen_d"],
        ascending=[True, False]
    )

    return stats_df, pairwise_df


def plot_class_counts(df):
    counts = df["class"].value_counts().reindex(TARGET_CLASSES).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.bar(counts.index, counts.values)
    ax.set_title("Distribusi Label")
    ax.set_ylabel("Jumlah Event")
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_counts.png"), bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df, features, stats_df, top_k=12):
    # Pilih fitur paling separable
    show_feats = stats_df["feature"].head(top_k).tolist()
    if not show_feats:
        return

    n = len(show_feats)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.8 * rows), dpi=140)
    axes = np.array(axes).reshape(-1)

    class_order = [c for c in TARGET_CLASSES if c in df["class"].unique()]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for idx, feat in enumerate(show_feats):
        ax = axes[idx]
        data = [df[df["class"] == c][feat].dropna().values for c in class_order]

        ax.boxplot(
            data,
            tick_labels=class_order,
            patch_artist=True,
            showmeans=True
        )

        for patch, color in zip(ax.artists, colors):
            patch.set_facecolor(color)

        pval = stats_df.loc[stats_df["feature"] == feat, "kruskal_p"].iloc[0]
        dmax = stats_df.loc[stats_df["feature"] == feat, "max_abs_cohen_d"].iloc[0]

        ax.set_title(f"{feat}\nKruskal p={pval:.2e} | max |d|={dmax:.2f}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)

    # Matikan subplot kosong
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top Feature Separability by Class", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "top_feature_boxplots.png"), bbox_inches="tight")
    plt.close(fig)


def plot_pca(df, features):
    X = df[features].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=42)

    X_imp = imputer.fit_transform(X)
    X_sc = scaler.fit_transform(X_imp)
    coords = pca.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

    class_order = [c for c in TARGET_CLASSES if c in df["class"].unique()]
    colors = {
        "Non-Event": "#4C78A8",
        "Pothole": "#F58518",
        "Speed Bump": "#54A24B",
    }

    for cls in class_order:
        mask = df["class"] == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.75,
            label=cls,
            color=colors.get(cls, None)
        )

    ax.set_title(
        f"PCA Separability Plot (Explained var: {pca.explained_variance_ratio_[0]:.2%} + "
        f"{pca.explained_variance_ratio_[1]:.2%})"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_scatter.png"), bbox_inches="tight")
    plt.close(fig)


def write_text_report(df, stats_df, pairwise_df):
    lines = []
    lines.append("# Feature Separability Analysis\n")
    lines.append("## Ringkasan Data\n")
    lines.append(df["class"].value_counts().reindex(TARGET_CLASSES).fillna(0).astype(int).to_string())
    lines.append("\n## Top Features by Separability\n")
    top = stats_df.head(10)[["feature", "kruskal_p", "max_abs_cohen_d", "mean_abs_cohen_d"]]
    lines.append(top.to_string(index=False))

    lines.append("\n## Interpretasi Cepat\n")
    lines.append("- Kruskal p kecil berarti distribusi fitur berbeda antar kelas.")
    lines.append("- |Cohen's d| besar berarti pemisahan antar kelas lebih kuat.")
    lines.append("- Fokuskan model pada fitur dengan p kecil dan effect size besar.\n")

    report_path = os.path.join(OUT_DIR, "separability_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # simpan tabel statistik
    stats_df.to_csv(os.path.join(OUT_DIR, "feature_stats.csv"), index=False)
    pairwise_df.to_csv(os.path.join(OUT_DIR, "pairwise_effect_sizes.csv"), index=False)


def main():
    df = load_and_merge()
    features = choose_available_features(df)

    print("Jumlah data per kelas:")
    print(df["class"].value_counts().reindex(TARGET_CLASSES).fillna(0).astype(int).to_string())

    stats_df, pairwise_df = compute_stats(df, features)

    # Simpan output tabular
    write_text_report(df, stats_df, pairwise_df)

    # Visualisasi
    plot_class_counts(df)
    plot_boxplots(df, features, stats_df, top_k=min(12, len(stats_df)))
    if len(features) >= 2:
        plot_pca(df, features)

    # Tampilkan ringkasan di console
    print("\nTop 10 fitur paling separable:")
    print(stats_df.head(10)[["feature", "kruskal_p", "max_abs_cohen_d", "mean_abs_cohen_d"]].to_string(index=False))
    print(f"\nHasil disimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()