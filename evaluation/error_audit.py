"""
Error Audit: False Positive Pothole Analysis
=============================================
Analyzes OOF predictions from both XGBoost and CNN to understand
why models over-predict Pothole on Non-Event samples.

Outputs:
  - evaluation/reports/error_audit_xgb_fp.csv
  - evaluation/reports/error_audit_summary.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

from config import XGB_OUT_DIR, CNN_OUT_DIR, get_logger

logger = get_logger(__name__)

XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def get_stratified_group_split(groups, y_raw, train_ratio=0.7):
    """Deterministic stratified group split — identical to train.py."""
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_idx = np.array([class_to_idx[val] for val in y_raw])

    group_names = np.unique(groups)
    group_counts = {g: np.array([np.sum(y_idx[groups == g] == i) for i in range(len(unique_classes))]) for g in group_names}
    total_counts = np.sum(list(group_counts.values()), axis=0)

    train_groups = set()
    test_groups = set()
    current_train = np.zeros(len(unique_classes))

    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)
    return list(train_groups), list(test_groups)


def audit_xgboost():
    """
    Re-run XGBoost 3-fold CV to collect OOF predictions with full metadata,
    then analyze False Positive Pothole patterns.
    """
    logger.info("=" * 60)
    logger.info("  XGBoost Error Audit: False Positive Pothole Analysis")
    logger.info("=" * 60)

    # --- Load data ---
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])

    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)

    df = df.dropna(subset=feature_cols)

    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp']
    X = df[feature_cols].values
    y_raw = df['label'].values
    groups = df['trip_id'].values
    source_values = df['source'].fillna('original').values if 'source' in df.columns else np.array(['original'] * len(df))

    # --- Split Dev/Test ---
    dev_groups_list, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev = X[dev_mask]
    y_dev_raw = y_raw[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]
    df_dev = df[dev_mask].reset_index(drop=True)

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    # --- 3-Fold CV to collect OOF predictions ---
    from xgboost import XGBClassifier

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_indices = []
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
        X_train, y_train = X_dev[train_idx], y_dev[train_idx]

        # Filter augmented from validation
        is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
        clean_val_idx = val_idx[is_original_val]
        X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

        # Inject missing classes
        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_dev == mc)[0][0]
                X_train = np.vstack([X_train, X_dev[global_idx]])
                y_train = np.append(y_train, mc)

        model = XGBClassifier(
            n_estimators=100, max_depth=4, min_child_weight=1,
            learning_rate=0.1, subsample=0.7, colsample_bytree=0.7,
            reg_lambda=1.0, reg_alpha=1.0, random_state=42, n_jobs=1
        )
        weights_train = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=weights_train)

        y_proba = model.predict_proba(X_val)
        y_pred = np.argmax(y_proba, axis=1)

        oof_indices.extend(clean_val_idx)
        oof_y_true.extend(y_val)
        oof_y_pred.extend(y_pred)
        oof_y_proba.extend(y_proba)

    oof_indices = np.array(oof_indices)
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)

    # --- Identify False Positives: Predicted Pothole, Actual Non-Event ---
    fp_mask = (oof_y_pred == p_idx) & (oof_y_true == non_event_idx)
    tp_mask = (oof_y_pred == p_idx) & (oof_y_true == p_idx)
    fn_mask = (oof_y_pred != p_idx) & (oof_y_true == p_idx)

    n_fp = fp_mask.sum()
    n_tp = tp_mask.sum()
    n_fn = fn_mask.sum()

    logger.info(f"OOF Pothole Predictions: TP={n_tp}, FP={n_fp}, FN={n_fn}")
    logger.info(f"OOF Precision Pothole: {n_tp / (n_tp + n_fp + 1e-9):.4f}")
    logger.info(f"OOF Recall Pothole: {n_tp / (n_tp + n_fn + 1e-9):.4f}")

    # --- Build FP DataFrame with metadata ---
    fp_df_indices = oof_indices[fp_mask]
    fp_records = []
    for i, dev_idx in enumerate(fp_df_indices):
        row = df_dev.iloc[dev_idx]
        proba = oof_y_proba[fp_mask][i]
        record = {
            'trip_id': row.get('trip_id', 'N/A'),
            'event_id': row.get('event_id', 'N/A'),
            'time_s': row.get('time_s', 'N/A'),
            'true_label': 'Non-Event',
            'pred_label': 'Pothole',
            'prob_NonEvent': proba[non_event_idx],
            'prob_Pothole': proba[p_idx],
            'prob_SpeedBump': proba[sb_idx] if sb_idx != -1 else 0.0,
        }
        # Add top discriminative features
        for feat in feature_cols:
            if feat in row.index:
                record[feat] = row[feat]
        fp_records.append(record)

    fp_df = pd.DataFrame(fp_records)

    # --- Similarly build TP DataFrame ---
    tp_df_indices = oof_indices[tp_mask]
    tp_records = []
    for i, dev_idx in enumerate(tp_df_indices):
        row = df_dev.iloc[dev_idx]
        proba = oof_y_proba[tp_mask][i]
        record = {
            'trip_id': row.get('trip_id', 'N/A'),
            'event_id': row.get('event_id', 'N/A'),
            'true_label': 'Pothole',
            'pred_label': 'Pothole',
            'prob_Pothole': proba[p_idx],
        }
        for feat in feature_cols:
            if feat in row.index:
                record[feat] = row[feat]
        tp_records.append(record)
    tp_df = pd.DataFrame(tp_records)

    # --- Save FP CSV ---
    fp_csv_path = os.path.join(REPORT_DIR, "error_audit_xgb_fp.csv")
    fp_df.to_csv(fp_csv_path, index=False)
    logger.info(f"FP detail saved to {fp_csv_path}")

    # --- Feature Distribution Comparison: TP vs FP ---
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("  XGBoost Error Audit: TP vs FP Feature Distribution (Pothole)")
    summary_lines.append("=" * 70)
    summary_lines.append(f"\nTotal TP: {n_tp} | Total FP: {n_fp} | Total FN: {n_fn}")
    summary_lines.append(f"OOF Precision: {n_tp / (n_tp + n_fp + 1e-9):.4f}")
    summary_lines.append(f"OOF Recall: {n_tp / (n_tp + n_fn + 1e-9):.4f}")
    summary_lines.append("")

    # FP per trip
    if not fp_df.empty:
        summary_lines.append("-" * 40)
        summary_lines.append("FP Distribution per Trip:")
        summary_lines.append("-" * 40)
        fp_per_trip = fp_df['trip_id'].value_counts()
        for trip, count in fp_per_trip.items():
            summary_lines.append(f"  {trip}: {count} FPs")
        summary_lines.append("")

    # Feature comparison
    KEY_FEATURES = [
        'peak_mag', 'peak_vertical', 'peak_mag_g', 'peak_vertical_g',
        'speed_mean', 'vert_jrk', 'asymmetry_score', 'vertical_energy',
        'max_jerk', 'peak_to_peak', 'kurtosis', 'crest_factor',
        'hjorth_activity', 'down_up_asymmetry', 'brake_to_bump_ratio',
        'horizontal_to_vertical_ratio', 'linear_jerk_3d_max',
        'impulse_factor', 'waveform_complexity', 'rise_time_ratio',
        'first_peak_polarity', 'snr_vertical'
    ]

    summary_lines.append("-" * 70)
    summary_lines.append(f"{'Feature':<35} {'TP Mean':>10} {'FP Mean':>10} {'Ratio FP/TP':>12} {'Signal'}")
    summary_lines.append("-" * 70)

    discriminative_features = []
    for feat in KEY_FEATURES:
        if feat in fp_df.columns and feat in tp_df.columns:
            tp_mean = tp_df[feat].mean()
            fp_mean = fp_df[feat].mean()
            ratio = fp_mean / (tp_mean + 1e-9)
            # Flag features where FP is very different from TP
            if ratio < 0.5 or ratio > 2.0:
                signal = "*** HIGH ***"
                discriminative_features.append((feat, tp_mean, fp_mean, ratio))
            elif ratio < 0.7 or ratio > 1.5:
                signal = "** MED **"
                discriminative_features.append((feat, tp_mean, fp_mean, ratio))
            else:
                signal = ""
            summary_lines.append(f"{feat:<35} {tp_mean:>10.4f} {fp_mean:>10.4f} {ratio:>12.3f} {signal}")

    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("  Most Discriminative Features (FP/TP ratio far from 1.0)")
    summary_lines.append("=" * 70)
    discriminative_features.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
    for feat, tp_mean, fp_mean, ratio in discriminative_features[:10]:
        direction = "FP >> TP" if ratio > 1.0 else "FP << TP"
        summary_lines.append(f"  {feat:<35} Ratio={ratio:.3f} ({direction})")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(REPORT_DIR, "error_audit_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved to {summary_path}")

    return fp_df, tp_df


def audit_cnn():
    """
    Use saved OOF predictions from CNN to analyze False Positive patterns.
    """
    logger.info("\n" + "=" * 60)
    logger.info("  1D-CNN Error Audit: False Positive Pothole Analysis")
    logger.info("=" * 60)

    # Load saved OOF predictions
    oof_y_true = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_true.npy"))
    oof_y_pred = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_pred.npy"))
    oof_y_proba = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_proba.npy"))

    classes = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_classes.npy"), allow_pickle=True)
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    fp_mask = (oof_y_pred == p_idx) & (oof_y_true == non_event_idx)
    tp_mask = (oof_y_pred == p_idx) & (oof_y_true == p_idx)
    fn_mask = (oof_y_pred != p_idx) & (oof_y_true == p_idx)

    n_fp = fp_mask.sum()
    n_tp = tp_mask.sum()
    n_fn = fn_mask.sum()

    logger.info(f"CNN OOF Pothole: TP={n_tp}, FP={n_fp}, FN={n_fn}")
    logger.info(f"CNN OOF Precision: {n_tp / (n_tp + n_fp + 1e-9):.4f}")
    logger.info(f"CNN OOF Recall: {n_tp / (n_tp + n_fn + 1e-9):.4f}")

    # Probability distribution analysis
    fp_probs = oof_y_proba[fp_mask][:, p_idx]
    tp_probs = oof_y_proba[tp_mask][:, p_idx]

    print(f"\n--- CNN Pothole Probability Distribution ---")
    print(f"  TP Pothole prob: mean={tp_probs.mean():.4f}, std={tp_probs.std():.4f}, "
          f"min={tp_probs.min():.4f}, max={tp_probs.max():.4f}")
    if n_fp > 0:
        print(f"  FP Pothole prob: mean={fp_probs.mean():.4f}, std={fp_probs.std():.4f}, "
              f"min={fp_probs.min():.4f}, max={fp_probs.max():.4f}")
        # How many FPs are borderline (prob < 0.6)?
        borderline = (fp_probs < 0.6).sum()
        confident = (fp_probs >= 0.6).sum()
        print(f"  FP borderline (<0.6): {borderline} | FP confident (>=0.6): {confident}")

    # Check what FP samples were actually predicted as (probability breakdown)
    if n_fp > 0:
        fp_all_probs = oof_y_proba[fp_mask]
        print(f"\n  FP avg probabilities: "
              f"NonEvent={fp_all_probs[:, non_event_idx].mean():.4f}, "
              f"Pothole={fp_all_probs[:, p_idx].mean():.4f}, "
              f"SpeedBump={fp_all_probs[:, sb_idx].mean():.4f}" if sb_idx != -1 else "")


if __name__ == "__main__":
    fp_df, tp_df = audit_xgboost()
    audit_cnn()
    print("\n[OK] Error audit selesai.")
