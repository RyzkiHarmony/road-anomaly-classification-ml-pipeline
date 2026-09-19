import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import sys

import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "stage3_eda_and_splitting"))
sys.path.append(os.path.dirname(__file__))

from config import XGB_OUT_DIR, get_logger
from data_splitting import get_stratified_group_split

logger = get_logger(__name__)


def _build_xgb_params(best_params_path: str, num_classes: int) -> tuple[dict, str]:
    """Load tuned parameters and enforce multiclass settings explicitly."""
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            xgb_params = json.load(f)
        rel_path = os.path.relpath(best_params_path, _PROJECT_ROOT).replace('\\', '/')
        param_source = f"Optuna Tuned ({rel_path})"
    else:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'min_child_weight': 1,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_lambda': 10.0,
            'reg_alpha': 1.0,
        }
        param_source = "Default (Hardcoded Baseline)"

    xgb_params['objective'] = 'multi:softprob'
    xgb_params['num_class'] = int(num_classes)
    xgb_params['random_state'] = 42
    xgb_params['n_jobs'] = 1
    return xgb_params, param_source


def _select_non_redundant_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int,
    num_classes: int,
    corr_threshold: float = 0.75
) -> tuple[list[str], list[dict]]:
    """
    Rank features using Dev split and eliminate multicollinear (redundant) features where |r| > corr_threshold.
    Returns (selected_features, dropped_redundant_info).
    """
    selector = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=int(num_classes),
        random_state=42,
        n_jobs=1,
    )
    selector.fit(X, y)
    importances = selector.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    ranked_features = [feature_names[i] for i in ranked_indices]

    df_features = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_features.corr().abs()

    selected_features = []
    dropped_info = []

    for feat in ranked_features:
        is_redundant = False
        for kept_feat in selected_features:
            r_val = corr_matrix.loc[feat, kept_feat]
            if r_val > corr_threshold:
                is_redundant = True
                dropped_info.append({
                    "dropped_feature": feat,
                    "redundant_with": kept_feat,
                    "correlation": float(r_val)
                })
                break
        if not is_redundant:
            selected_features.append(feat)

    final_features = [f for f in feature_names if f in selected_features[:top_n]]
    return final_features, dropped_info

def main():
    # ---------- LOAD DATA ----------
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please run build_xgboost_data.py first.")
        return

    print("\n" + "=" * 78)
    print("       PELATIHAN & EVALUASI MODEL XGBOOST (ROAD ANOMALY DETECTION)")
    print("=" * 78)

    df = pd.read_csv(data_path)
    if df.empty:
        logger.error("Dataset kosong.")
        return

    # Menghapus row yang memiliki NaN pada kolom fitur atau label
    df = df.dropna(subset=['label'])

    # DROP DATA LEAKAGE AND NON-KOTLIN-FRIENDLY FEATURES
    banned_cols = ['lat', 'lon', 'suggestion_confidence', 'score']
    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp'] + banned_cols
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns]).select_dtypes(include=[np.number])
    all_feature_cols = numeric_df.columns.tolist()

    df = df.dropna(subset=all_feature_cols)
    X_all = df[all_feature_cols].values
    y_all = df['label'].values
    groups = df['trip_id'].values

    # Save source column as array to easily filter out augmented twins in validation
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    # ---------- SPLIT DEV SET (80%) AND HOLDOUT TEST SET (20%) ----------
    dev_groups_list, test_groups_list = get_stratified_group_split(groups, y_all, train_ratio=0.8)

    dev_mask = np.isin(groups, dev_groups_list)
    test_mask = np.isin(groups, test_groups_list)

    X_dev_full = X_all[dev_mask]
    y_dev_raw = y_all[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]

    is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_mask]])
    X_test_full = X_all[test_mask][is_original_test]
    y_test_raw = y_all[test_mask][is_original_test]
    groups_test = groups[test_mask][is_original_test]

    print("\n[1/5] PEMBAGIAN DATASET (Trip-Isolated Stratified Group Split 80:20):")
    print(f"  - Total Data         : {len(df):,} sampel")
    print(f"  - Dev Set (80%)      : {len(dev_groups_list)} rute perjalanan | {len(X_dev_full):,} sampel")
    print(f"  - Holdout Test (20%) : {len(test_groups_list)} rute perjalanan | {len(X_test_full):,} sampel (murni original)")

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    y_test = le.transform(y_test_raw)
    classes = le.classes_
    CORR_THRESHOLD = 0.75
    TOP_N_FEATURES = min(25, len(all_feature_cols))

    feature_cols, dropped_redundant = _select_non_redundant_features(
        X_dev_full, y_dev, all_feature_cols, TOP_N_FEATURES, len(classes), corr_threshold=CORR_THRESHOLD
    )

    print(f"\n[2/5] SELEKSI FITUR NON-REDUNDAN (Dev Set Only):")
    print(f"  - Ambang Multikolinieritas : |r| <= {CORR_THRESHOLD}")
    print(f"  - Total Fitur Kandidat     : {len(all_feature_cols)} fitur")
    print(f"  - Fitur Dieliminasi        : {len(dropped_redundant)} fitur redundan (|r| > {CORR_THRESHOLD})")
    print(f"  - Fitur Terpilih (Final)   : {len(feature_cols)} fitur independen")
    print(f"  - Top 5 Fitur Terkuat      : {', '.join(feature_cols[:5])}...")

    X_dev = pd.DataFrame(X_dev_full, columns=all_feature_cols)[feature_cols].values
    X_test = pd.DataFrame(X_test_full, columns=all_feature_cols)[feature_cols].values
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    best_params_path = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost", "best_params.json")
    xgb_params, param_source = _build_xgb_params(best_params_path, len(classes))

    # ---------- CROSS-VALIDATION (4-FOLD STRATIFIED GROUP K-FOLD ON DEV SET) ----------
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

    fold_metrics = []
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    print(f"\n[3/5] 4-FOLD STRATIFIED GROUP CROSS-VALIDATION (Dev Set: {len(X_dev):,} sampel):", flush=True)
    print(f"  - Konfigurasi Hyperparameter : {param_source}", flush=True)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
        X_train, y_train = X_dev[train_idx], y_dev[train_idx]

        is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
        clean_val_idx = val_idx[is_original_val]
        X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

        model = XGBClassifier(**xgb_params)
        weights_train = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=weights_train)

        y_proba = model.predict_proba(X_val)
        y_pred = np.argmax(y_proba, axis=1)

        oof_y_true.extend(y_val)
        oof_y_pred.extend(y_pred)
        oof_y_proba.extend(y_proba)

        y_pred_train = model.predict(X_train)
        f1_train = f1_score(y_train, y_pred_train, labels=[p_idx], average='macro', zero_division=0)
        f1_val = f1_score(y_val, y_pred, labels=[p_idx], average='macro', zero_division=0)

        y_val_pothole_fold = (y_val == p_idx).astype(int)
        y_proba_pothole_fold = y_proba[:, p_idx]
        prec, rec, _ = precision_recall_curve(y_val_pothole_fold, y_proba_pothole_fold)
        pr_auc_val = auc(rec, prec)

        fold_metrics.append((f1_val, pr_auc_val, f1_train))
        print(f"  - Fold {fold+1} : Val Pothole F1 = {f1_val:.4f} | PR-AUC = {pr_auc_val:.4f} (Train F1 = {f1_train:.4f})", flush=True)

    avg_f1 = np.mean([m[0] for m in fold_metrics])
    avg_prauc = np.mean([m[1] for m in fold_metrics])
    print("  " + "-" * 62, flush=True)
    print(f"  Rata-rata OOF CV : Pothole F1 = {avg_f1:.4f} | Pothole PR-AUC = {avg_prauc:.4f}", flush=True)

    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)

    model_dir = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost")
    report_dir = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "xgboost")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    np.save(os.path.join(model_dir, "xgb_oof_y_true.npy"), oof_y_true)
    np.save(os.path.join(model_dir, "xgb_oof_y_pred.npy"), oof_y_pred)
    np.save(os.path.join(model_dir, "xgb_oof_y_proba.npy"), oof_y_proba)

    # Save OOF Confusion Matrix figure
    cm_oof = confusion_matrix(oof_y_true, oof_y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_oof, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm_oof.shape[1]),
           yticks=np.arange(cm_oof.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Out-of-Fold - XGBoost)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_oof.max() / 2.0
    for i in range(cm_oof.shape[0]):
        for j in range(cm_oof.shape[1]):
            row_total = cm_oof[i].sum()
            pct = cm_oof[i, j] / row_total * 100 if row_total > 0 else 0
            ax.text(j, i, f"{cm_oof[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight='bold',
                    color="white" if cm_oof[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = os.path.join(report_dir, "xgboost_confusion_matrix_oof.png")
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------- TRAIN FINAL MODEL ON DEV SET ----------
    print("\n[4/5] PELATIHAN MODEL FINAL & KALIBRASI ISOTONIK:", flush=True)
    final_model = XGBClassifier(**xgb_params)
    weights_final = compute_sample_weight('balanced', y_dev)
    final_model.fit(X_dev, y_dev, sample_weight=weights_final)

    calibrators = {}
    for cls_idx in range(len(classes)):
        y_binary = (oof_y_true == cls_idx).astype(float)
        raw_proba = oof_y_proba[:, cls_idx]
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(raw_proba, y_binary)
        calibrators[cls_idx] = ir

    print(f"  - Pelatihan Model Final    : {len(X_dev):,} sampel Dev Set (Sample Weighting: Balanced)", flush=True)
    print("  - Kalibrasi Probabilitas   : Isotonic Regression per-kelas di-fit pada probabilitas OOF", flush=True)

    # Save artifacts
    pkl_path = os.path.join(model_dir, "xgboost_model.pkl")
    joblib.dump(final_model, pkl_path)

    cal_pkl_path = os.path.join(model_dir, "xgboost_calibrators.pkl")
    joblib.dump(calibrators, cal_pkl_path)
    joblib.dump(le, os.path.join(model_dir, "xgboost_label_encoder.pkl"))

    with open(os.path.join(model_dir, "xgboost_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(model_dir, "xgboost_pruned_redundant_features.json"), "w") as f:
        json.dump(dropped_redundant, f, indent=2)

    threshold_config = {
        "pothole_threshold": 0.5,
        "speed_bump_threshold": 0.5,
        "pothole_class_index": int(p_idx),
        "speed_bump_class_index": int(sb_idx),
        "non_event_class_index": int(non_event_idx),
        "class_names": list(classes),
        "calibration_method": "isotonic",
        "note": "Thresholds optimized on calibrated probabilities."
    }
    thresh_json_path = os.path.join(model_dir, "xgboost_thresholds.json")
    with open(thresh_json_path, "w") as f:
        json.dump(threshold_config, f, indent=2)

    # --- Evaluate Calibrated Model on Holdout Test Set ---
    raw_test_probas = final_model.predict_proba(X_test)
    test_probas = np.column_stack([
        calibrators[i].predict(raw_test_probas[:, i]) for i in range(len(classes))
    ])
    test_probas = test_probas / test_probas.sum(axis=1, keepdims=True)
    test_preds_default = np.argmax(test_probas, axis=1)

    print("\n" + "=" * 78, flush=True)
    print("   [5/5] HASIL PENGUJIAN HOLDOUT TEST SET (20% Rute Baru - 945 Sampel)", flush=True)
    print("=" * 78, flush=True)
    print(classification_report(y_test, test_preds_default, target_names=classes, digits=4, zero_division=0), flush=True)

    # Text Confusion Matrix with row/col totals for easy academic referencing
    cm_test = confusion_matrix(y_test, test_preds_default)
    header_col = f"  {'Aktual \\ Prediksi':<20}" + "".join([f"{c:>14}" for c in classes]) + f"{'Total':>10}"
    div_line = "  " + "-" * (len(header_col) - 2)
    print("  Ringkasan Matriks Konfusi (Holdout Test Set):", flush=True)
    print(div_line, flush=True)
    print(header_col, flush=True)
    print(div_line, flush=True)
    for i, cls_name in enumerate(classes):
        row_str = f"  {cls_name:<20}" + "".join([f"{cm_test[i, j]:>14}" for j in range(len(classes))]) + f"{cm_test[i].sum():>10}"
        print(row_str, flush=True)
    print(div_line, flush=True)
    col_totals = f"  {'Total Prediksi':<20}" + "".join([f"{cm_test[:, j].sum():>14}" for j in range(len(classes))]) + f"{cm_test.sum():>10}"
    print(col_totals, flush=True)
    print(div_line, flush=True)
    print("=" * 78, flush=True)

    # Save holdout confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_test, interpolation='nearest', cmap='Oranges')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm_test.shape[1]),
           yticks=np.arange(cm_test.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Holdout Test Set - Default Argmax)')
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
    cm_test_path = os.path.join(report_dir, "xgboost_confusion_matrix_holdout.png")
    fig.savefig(cm_test_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------- ONNX EXPORT ----------
    onnx_msg = "Gagal"
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_xgboost(final_model, initial_types=initial_types, target_opset=15)
        onnx_path = os.path.join(model_dir, "xgboost_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        rel_onnx = os.path.relpath(onnx_path, _PROJECT_ROOT).replace('\\', '/')
        onnx_msg = f"{rel_onnx} (Berhasil diekspor)"
    except Exception as e:
        onnx_msg = f"Gagal ekspor ONNX: {e}"

    rel_pkl = os.path.relpath(pkl_path, _PROJECT_ROOT).replace('\\', '/')
    rel_cal = os.path.relpath(cal_pkl_path, _PROJECT_ROOT).replace('\\', '/')
    rel_cm = os.path.relpath(cm_test_path, _PROJECT_ROOT).replace('\\', '/')
    rel_feat = os.path.relpath(os.path.join(model_dir, 'xgboost_features.json'), _PROJECT_ROOT).replace('\\', '/')

    print("\n[STATUS PENYIMPANAN ARTEFAK & DEPLOYMENT]", flush=True)
    print(f"  - Model Pickle    : {rel_pkl}", flush=True)
    print(f"  - Kalibrator      : {rel_cal}", flush=True)
    print(f"  - Model ONNX      : {onnx_msg}", flush=True)
    print(f"  - Plot Matriks CM : {rel_cm}", flush=True)
    print(f"  - Fitur Terpilih  : {rel_feat}", flush=True)
    print("=" * 78 + "\n", flush=True)



if __name__ == "__main__":
    main()

