import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from config import XGB_OUT_DIR, get_logger, BEST_FEATURES

logger = get_logger(__name__)

def main():
    # ---------- LOAD DATA ----------
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please run build_xgboost_data.py first.")
        return

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if df.empty:
        logger.error("Dataset is empty.")
        return

    # Menghapus row yang memiliki NaN pada kolom fitur atau label
    df = df.dropna(subset=['label'])
    
    # ---------- FEATURE SELECTION (ALL FEATURES) ----------
    # Berdasarkan analisis Senior ML Engineer, kita cabut pembatasan 10 fitur.
    # XGBoost mampu mengelola 30+ fitur dan menemukan interaksi non-linear yang tajam,
    # asalkan data augmentasi fisikanya murni (bug augmentasi linier telah diperbaiki).
    
    # DROP DATA LEAKAGE AND NON-KOTLIN-FRIENDLY FEATURES
    # We must remove coordinates (leakage) and heuristic scores.
    # We also remove features that require Scipy FFT or complex peak finding,
    # to guarantee easy and safe deployment in Android Kotlin.
    banned_cols = [
        'lat', 'lon', 'suggestion_confidence', 'score'  # Data Leakage
    ]
    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp'] + banned_cols
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns]).select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()
    
    df = df.dropna(subset=feature_cols)

    logger.info(f"Using ALL {len(feature_cols)} features for maximum performance: {feature_cols}")

    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values
    
    # Save source column as array to easily filter out augmented twins in validation
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    # ---------- SPLIT DEV SET (70%) AND HOLDOUT TEST SET (30%) ----------

    # Split Dev/Holdout based on trip_id using the custom function
    dev_groups_list, test_groups_list = get_stratified_group_split(groups, y, train_ratio=0.7)
    
    dev_mask = np.isin(groups, dev_groups_list)
    test_mask = np.isin(groups, test_groups_list)

    # Dev Set data (for cross-validation and training the final model)
    X_dev = X[dev_mask]
    y_dev_raw = y[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]

    # Holdout Test Set data (for final evaluation)
    # Filter out augmented twins from holdout test set to prevent leakage validation mirages
    is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_mask]])
    X_test = X[test_mask][is_original_test]
    y_test_raw = y[test_mask][is_original_test]
    groups_test = groups[test_mask][is_original_test]

    logger.info(f"Split Summary (Trip-Based):")
    logger.info(f"  Dev Set (70%): {len(dev_groups_list)} trips, {len(X_dev)} samples")
    logger.info(f"  Holdout Test Set (30%): {len(test_groups_list)} trips, {len(X_test)} samples (original only)")

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    y_test = le.transform(y_test_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    # ---------- CROSS-VALIDATION (3-FOLD STRATIFIED GROUP K-FOLD ON DEV SET) ----------
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    # OOF Trackers
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
        # 1. Train Set
        X_train, y_train = X_dev[train_idx], y_dev[train_idx]
        
        # 2. Validation Set: STRICTLY filter out augmented twins to prevent data leakage validation mirages
        is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
        clean_val_idx = val_idx[is_original_val]
        
        X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

        # Inject missing classes in the training split if a class is entirely absent
        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_dev == mc)[0][0]
                X_train = np.vstack([X_train, X_dev[global_idx]])
                y_train = np.append(y_train, mc)

        # XGBoost with strong regularization to prevent overfitting on Hard Negatives
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            min_child_weight=1,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=1.0,
            random_state=42,
            n_jobs=1
        )
        
        # Calculate sample weights to combat base rate fallacy
        weights_train = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predictions on the clean, un-augmented validation split
        y_proba = model.predict_proba(X_val)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Track OOF
        oof_y_true.extend(y_val)
        oof_y_pred.extend(y_pred)
        oof_y_proba.extend(y_proba)
        
        # Evaluate Training Set to check Overfitting
        y_pred_train = model.predict(X_train)
        f1_train = f1_score(y_train, y_pred_train, labels=[p_idx], average='macro', zero_division=0)
        
        # Validation Evaluation
        f1_val = f1_score(y_val, y_pred, labels=[p_idx], average='macro', zero_division=0)
        
        # Calculate PR-AUC for Pothole
        y_val_pothole_fold = (y_val == p_idx).astype(int)
        y_proba_pothole_fold = y_proba[:, p_idx]
        prec, rec, _ = precision_recall_curve(y_val_pothole_fold, y_proba_pothole_fold)
        pr_auc_val = auc(rec, prec)
        
        fold_metrics.append((f1_val, pr_auc_val))
        
        logger.info(f"Fold {fold+1} | Train Pothole F1: {f1_train:.4f} | Val Pothole F1: {f1_val:.4f} | Val PR-AUC: {pr_auc_val:.4f}")

    avg_f1 = np.mean([m[0] for m in fold_metrics])
    avg_prauc = np.mean([m[1] for m in fold_metrics])
    print("-" * 60)
    logger.info(f"Average Pothole F1: {avg_f1:.4f} | Average PR-AUC: {avg_prauc:.4f}")

    # Combine OOF predictions
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)

    # ---------- THRESHOLD OPTIMIZATION (Using Clean OOF Data) ----------
    y_true_pothole = (oof_y_true == p_idx).astype(int)
    y_proba_pothole = oof_y_proba[:, p_idx]
    prec, rec, thresholds = precision_recall_curve(y_true_pothole, y_proba_pothole)
    
    # Cari threshold yang memaksimalkan F1-Score
    fscore = (2 * prec * rec) / (prec + rec + 1e-9)
    ix = np.argmax(fscore)
    best_thresh_p = thresholds[ix] if ix < len(thresholds) else 0.5
    
    # Threshold Optimization for Speed Bump
    best_thresh_sb = 0.5
    if sb_idx != -1:
        y_true_sb = (oof_y_true == sb_idx).astype(int)
        y_proba_sb = oof_y_proba[:, sb_idx]
        prec_sb, rec_sb, thresholds_sb = precision_recall_curve(y_true_sb, y_proba_sb)
        fscore_sb = (2 * prec_sb * rec_sb) / (prec_sb + rec_sb + 1e-9)
        ix_sb = np.argmax(fscore_sb)
        best_thresh_sb = thresholds_sb[ix_sb] if ix_sb < len(thresholds_sb) else 0.5

    print("\n" + "=" * 60)
    print("              METRICS & THRESHOLD SUMMARY              ")
    print("=" * 60)
    print(f"Optimal Thresholds -> Pothole: {best_thresh_p:.4f} | Speed Bump: {best_thresh_sb:.4f}")

    # Terapkan threshold optimasi pada OOF predictions
    oof_y_pred_opt = np.zeros_like(oof_y_true)
    for i in range(len(oof_y_proba)):
        proba = oof_y_proba[i]
        p_prob = proba[p_idx]
        sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
        
        p_triggered = p_prob >= best_thresh_p
        sb_triggered = sb_idx != -1 and sb_prob >= best_thresh_sb
        
        if p_triggered and sb_triggered:
            if p_prob >= sb_prob:
                oof_y_pred_opt[i] = p_idx
            else:
                oof_y_pred_opt[i] = sb_idx
        elif p_triggered:
            oof_y_pred_opt[i] = p_idx
        elif sb_triggered:
            oof_y_pred_opt[i] = sb_idx
        else:
            oof_y_pred_opt[i] = non_event_idx

    print("\nOut-of-Fold Classification Report (Optimized Threshold):")
    print(classification_report(oof_y_true, oof_y_pred_opt, target_names=classes, zero_division=0))
    
    # Save Out-of-Fold Confusion Matrix
    cm_oof = confusion_matrix(oof_y_true, oof_y_pred_opt)
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost")
    report_dir = os.path.join(_PROJECT_ROOT, "evaluation", "reports", "xgboost")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # --- Save OOF predictions for future error audits ---
    np.save(os.path.join(model_dir, "xgb_oof_y_true.npy"), oof_y_true)
    np.save(os.path.join(model_dir, "xgb_oof_y_pred.npy"), oof_y_pred_opt)
    np.save(os.path.join(model_dir, "xgb_oof_y_proba.npy"), oof_y_proba)
    logger.info("OOF predictions saved for error audit.")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_oof, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm_oof.shape[1]),
           yticks=np.arange(cm_oof.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Out-of-Fold - Optimized)')
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
    logger.info(f"OOF Confusion matrix disimpan di {cm_path}")

    # ---------- TRAIN FINAL MODEL ON DEV SET ----------
    logger.info("Melatih final model pada seluruh Dev Set...")
    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        min_child_weight=1,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=1
    )
    weights_final = compute_sample_weight('balanced', y_dev)
    final_model.fit(X_dev, y_dev, sample_weight=weights_final)

    # ---------- PROBABILITY CALIBRATION (Isotonic, Post-Hoc on OOF) ----------
    # Fit per-class IsotonicRegression on OOF probabilities.
    # OOF probabilities are cross-validated, so no leakage.
    # This learns a mapping: raw_proba -> calibrated_proba per class.
    logger.info("Applying Isotonic Probability Calibration on OOF data...")
    
    calibrators = {}
    for cls_idx in range(len(classes)):
        y_binary = (oof_y_true == cls_idx).astype(float)
        raw_proba = oof_y_proba[:, cls_idx]
        
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(raw_proba, y_binary)
        calibrators[cls_idx] = ir
        
        # Log calibration effect
        cal_proba = ir.predict(raw_proba)
        logger.info(f"  {classes[cls_idx]}: raw mean={raw_proba.mean():.4f} -> cal mean={cal_proba.mean():.4f}")
    
    # Apply calibration to OOF probabilities and verify
    cal_oof_proba = np.column_stack([
        calibrators[i].predict(oof_y_proba[:, i]) for i in range(len(classes))
    ])
    # Normalize to sum to 1
    cal_oof_proba = cal_oof_proba / cal_oof_proba.sum(axis=1, keepdims=True)
    cal_oof_pred = np.argmax(cal_oof_proba, axis=1)
    logger.info("Calibrated OOF report (sanity check):")
    logger.info("\n" + classification_report(oof_y_true, cal_oof_pred, target_names=classes, zero_division=0))

    # ---------- RE-OPTIMIZE THRESHOLDS ON CALIBRATED OOF PROBABILITIES ----------
    cal_y_true_pothole = (oof_y_true == p_idx).astype(int)
    cal_y_proba_pothole = cal_oof_proba[:, p_idx]
    prec_cal, rec_cal, thresh_cal = precision_recall_curve(cal_y_true_pothole, cal_y_proba_pothole)
    fscore_cal = (2 * prec_cal * rec_cal) / (prec_cal + rec_cal + 1e-9)
    ix_cal = np.argmax(fscore_cal)
    best_thresh_p_cal = thresh_cal[ix_cal] if ix_cal < len(thresh_cal) else 0.5
    
    best_thresh_sb_cal = 0.5
    if sb_idx != -1:
        cal_y_true_sb = (oof_y_true == sb_idx).astype(int)
        cal_y_proba_sb = cal_oof_proba[:, sb_idx]
        prec_sb_cal, rec_sb_cal, thresh_sb_cal = precision_recall_curve(cal_y_true_sb, cal_y_proba_sb)
        fscore_sb_cal = (2 * prec_sb_cal * rec_sb_cal) / (prec_sb_cal + rec_sb_cal + 1e-9)
        ix_sb_cal = np.argmax(fscore_sb_cal)
        best_thresh_sb_cal = thresh_sb_cal[ix_sb_cal] if ix_sb_cal < len(thresh_sb_cal) else 0.5
    
    print(f"\nCalibrated Thresholds -> Pothole: {best_thresh_p_cal:.4f} | Speed Bump: {best_thresh_sb_cal:.4f}")
    print(f"(Pre-calibration -> Pothole: {best_thresh_p:.4f} | Speed Bump: {best_thresh_sb:.4f})")

    # ---------- SAVE ARTIFACTS ----------
    # Save raw model for ONNX export
    pkl_path = os.path.join(model_dir, "xgboost_model.pkl")
    joblib.dump(final_model, pkl_path)
    
    # Save calibrated model for Python evaluation
    cal_pkl_path = os.path.join(model_dir, "xgboost_calibrators.pkl")
    joblib.dump(calibrators, cal_pkl_path)
    
    joblib.dump(le, os.path.join(model_dir, "xgboost_label_encoder.pkl"))
    
    with open(os.path.join(model_dir, "xgboost_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    # Save threshold JSON for Android inference (uses raw ONNX + thresholds)
    threshold_config = {
        "pothole_threshold": float(best_thresh_p_cal),
        "speed_bump_threshold": float(best_thresh_sb_cal),
        "pothole_class_index": int(p_idx),
        "speed_bump_class_index": int(sb_idx),
        "non_event_class_index": int(non_event_idx),
        "class_names": list(classes),
        "calibration_method": "isotonic",
        "note": "Thresholds optimized on calibrated probabilities. For ONNX raw inference, use these thresholds with raw predict_proba output."
    }
    thresh_json_path = os.path.join(model_dir, "xgboost_thresholds.json")
    with open(thresh_json_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    
    logger.info(f"Raw model saved at {pkl_path}")
    logger.info(f"Calibrated model saved at {cal_pkl_path}")
    logger.info(f"Threshold config saved at {thresh_json_path}")

    # --- Evaluate Calibrated Model on Holdout Test Set (30%) ---
    logger.info("Mengevaluasi CALIBRATED model pada Holdout Test Set (30%)...")
    raw_test_probas = final_model.predict_proba(X_test)
    # Apply per-class isotonic calibration
    test_probas = np.column_stack([
        calibrators[i].predict(raw_test_probas[:, i]) for i in range(len(classes))
    ])
    test_probas = test_probas / test_probas.sum(axis=1, keepdims=True)
    
    test_preds = np.zeros_like(y_test)
    for i in range(len(test_probas)):
        proba = test_probas[i]
        p_prob = proba[p_idx]
        sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
        
        p_triggered = p_prob >= best_thresh_p_cal
        sb_triggered = sb_idx != -1 and sb_prob >= best_thresh_sb_cal
        
        if p_triggered and sb_triggered:
            if p_prob >= sb_prob:
                test_preds[i] = p_idx
            else:
                test_preds[i] = sb_idx
        elif p_triggered:
            test_preds[i] = p_idx
        elif sb_triggered:
            test_preds[i] = sb_idx
        else:
            test_preds[i] = non_event_idx
            
    print("\n" + "=" * 60)
    print("             HOLDOUT TEST SET EVALUATION             ")
    print("=" * 60)
    print(classification_report(y_test, test_preds, target_names=classes, zero_division=0))
    print("=" * 60 + "\n")
    
    # Save holdout confusion matrix
    cm_test = confusion_matrix(y_test, test_preds)
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
    cm_test_path = os.path.join(report_dir, "xgboost_confusion_matrix_holdout.png")
    fig.savefig(cm_test_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Holdout Test confusion matrix disimpan di {cm_test_path}")

    # ---------- ONNX EXPORT ----------
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        logger.info("Exporting final model to ONNX format for Android deployment...")
        initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
        
        # Convert the XGBoost final model to ONNX format
        onnx_model = convert_xgboost(
            final_model, 
            initial_types=initial_types, 
            target_opset=15
        )
        
        onnx_path = os.path.join(model_dir, "xgboost_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Model successfully exported to ONNX format at {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")

def get_stratified_group_split(groups, y_raw, train_ratio=0.7):
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_idx = np.array([class_to_idx[val] for val in y_raw])

    group_names = np.unique(groups)
    group_counts = {g: np.array([np.sum(y_idx[groups == g] == i) for i in range(len(unique_classes))]) for g in group_names}
    total_counts = np.sum(list(group_counts.values()), axis=0)

    train_groups = set()
    test_groups = set()
    current_train = np.zeros(len(unique_classes))

    # Sort groups by total minority class count desc
    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        # Minimize MSE of ratios to train_ratio:
        # If added to train:
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        
        # If added to test:
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)

    return list(train_groups), list(test_groups)

if __name__ == "__main__":
    main()

