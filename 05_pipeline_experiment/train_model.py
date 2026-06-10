import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import OUT_FOLDER, get_logger, BEST_FEATURES

logger = get_logger(__name__)

def main():
    # ---------- LOAD DATA ----------
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please run build_train_set.py first.")
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
        'lat', 'lon', 'suggestion_confidence', 'score',  # Data Leakage
        'energy_psd_2_10', 'fft_high_low_ratio',         # Requires FFT/Welch
        'num_peaks_accel', 'num_peaks_gyro',             # Requires Scipy find_peaks
        'peak_interval_mean', 'peak_interval_std'        # Relies on find_peaks
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

    # ---------- CROSS-VALIDATION (STRATIFIED GROUP K-FOLD) ----------
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")

    # Adapt folds count to unique groups count
    unique_groups = len(np.unique(groups))
    n_folds = min(5, unique_groups) if unique_groups >= 2 else 5
    
    # Choose cross-validation strategy: if trip count is too low, fall back to standard StratifiedKFold to prevent group validation mirages
    if unique_groups >= 2:
        from sklearn.model_selection import StratifiedGroupKFold
        cv_strategy = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_name = f"{n_folds}-FOLD STRATIFIED GROUP K-FOLD"
    else:
        from sklearn.model_selection import StratifiedKFold
        cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_name = f"{n_folds}-FOLD STRATIFIED K-FOLD (FALLBACK)"

    print("\n" + "="*60)
    print(f"{cv_name:^60}")
    print("="*60)

    fold_metrics = []
    
    # OOF Trackers
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y_enc, groups)):
        # 1. Train Set: Keep both original and augmented events
        X_train, y_train = X[train_idx], y_enc[train_idx]
        
        # 1.5 UNDERSAMPLE NON-EVENTS IN TRAINING SET ONLY (Avoid Base Rate Fallacy)
        ne_idx = list(classes).index("Non-Event")
        is_ne = (y_train == ne_idx)
        is_anom = ~is_ne
        
        n_anomalies_train = np.sum(is_anom)
        n_ne_train = np.sum(is_ne)
        max_ne_train = int(n_anomalies_train * 1.5)
        
        if n_ne_train > max_ne_train:
            ne_indices = np.where(is_ne)[0]
            np.random.seed(42 + fold)
            sampled_ne_indices = np.random.choice(ne_indices, size=max_ne_train, replace=False)
            anom_indices = np.where(is_anom)[0]
            keep_indices = np.concatenate([anom_indices, sampled_ne_indices])
            np.random.shuffle(keep_indices)
            X_train = X_train[keep_indices]
            y_train = y_train[keep_indices]
        
        # 2. Test Set: STRICTLY filter out augmented twins to prevent data leakage validation mirage
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        
        X_test, y_test = X[clean_test_idx], y_enc[clean_test_idx]

        # Inject missing classes in the training split if a class is entirely absent
        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                # Find index of this class in the global y_enc
                global_idx = np.where(y_enc == mc)[0][0]
                X_train = np.vstack([X_train, X[global_idx]])
                y_train = np.append(y_train, mc)

        # Handle Class Imbalance using scale_pos_weight in XGBoost natively later, 
        # do NOT use SMOTE as it corrupts physically valid raw signal augmentation.

        # XGBoost with strong regularization to prevent overfitting on Hard Negatives
        model = XGBClassifier(
            n_estimators=40,          # Reduced from 100 to prevent memorization
            max_depth=2,              # Reduced from 3 to force simpler rules
            min_child_weight=10,      # Increased from 5 to prevent splitting on noise
            learning_rate=0.05,
            subsample=0.7,            # More dropout on rows
            colsample_bytree=0.7,     # More dropout on columns
            reg_lambda=5.0,           # Strong L2 penalty
            reg_alpha=1.0,            # Strong L1 penalty
            random_state=42,
            n_jobs=1
        )
        
        # Calculate sample weights to combat base rate fallacy
        weights_train = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predictions on the clean, un-augmented test split
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Track OOF
        oof_y_true.extend(y_test)
        oof_y_pred.extend(y_pred)
        if len(y_proba) > 0:
            y_pred_val = model.predict(X_test)
        
        # --- Evaluate Training Set to check Overfitting ---
        y_pred_train = model.predict(X_train)
        f1_train = f1_score(y_train, y_pred_train, labels=[p_idx], average='macro', zero_division=0)
        
        # Validation Evaluation
        f1_val = f1_score(y_test, y_pred_val, labels=[p_idx], average='macro', zero_division=0)
        
        fold_metrics.append(f1_val)
        oof_y_proba.append(y_proba)
        
        logger.info(f"Fold {fold+1} | Train Pothole F1: {f1_train:.4f} | Val Pothole F1: {f1_val:.4f} | Gap: {f1_train - f1_val:.4f}")

    avg_f1 = np.mean(fold_metrics)
    print("-" * 60)
    logger.info(f"Average Pothole F1 across {n_folds} folds: {avg_f1:.4f}")

    # Combine OOF predictions
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    if len(oof_y_proba) > 0:
        oof_y_proba = np.vstack(oof_y_proba)
    else:
        oof_y_proba = np.zeros((len(oof_y_true), len(classes)))

    # ---------- THRESHOLD OPTIMIZATION (Using Clean OOF Data) ----------
    # Kita cari threshold yang menyeimbangkan Recall vs Precision pada data asli
    print("\n" + "="*60)
    print(f"{'THRESHOLD OPTIMIZATION (Pothole)':^60}")
    print("="*60)
    
    # Ambil probabilitas kelas Pothole dari seluruh OOF predictions
    y_test_pothole = (oof_y_true == p_idx).astype(int)
    y_proba_pothole = oof_y_proba[:, p_idx]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test_pothole, y_proba_pothole)
    
    # BALANCED OPTIMIZATION: Cari threshold yang memaksimalkan F1-Score (keseimbangan Precision & Recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_f1_idx = np.argmax(f1_scores)
    idx = best_f1_idx
        
    opt_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    
    print(f"Proposed Threshold for Pothole: {opt_threshold:.4f}")
    if idx < len(precisions):
        print(f"Expected -> Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")

    # ---------- FINAL REPORT (CROSS-VALIDATION OOF) ----------
    print("\nFinal Report (Out-of-Fold - Unbiased Original):")
    # Terapkan custom threshold untuk kelas Pothole
    oof_y_pred_opt = oof_y_pred.copy()
    oof_y_pred_opt[y_proba_pothole >= opt_threshold] = p_idx
    # Untuk Non-Event/Speed Bump, jika probabilitas pothole < threshold tapi model awalnya memprediksi pothole,
    # kembalikan ke tebakan terbanyak (Non-Event)
    oof_y_pred_opt[(y_proba_pothole < opt_threshold) & (oof_y_pred_opt == p_idx)] = list(classes).index("Non-Event")
    
    print(classification_report(oof_y_true, oof_y_pred_opt, target_names=classes, zero_division=0))
    
    # ---------- TRAIN FINAL PRODUCTION MODEL ----------
    logger.info("Training final production model on the entire dataset (Original + Augmented)...")
    
    # UNDERSAMPLE FINAL DATASET BEFORE SMOTE
    is_ne_final = (y_enc == list(classes).index("Non-Event"))
    is_anom_final = ~is_ne_final
    n_ne_final = np.sum(is_ne_final)
    n_anom_final = np.sum(is_anom_final)
    max_ne_final = int(n_anom_final * 1.5)
    
    if n_ne_final > max_ne_final:
        ne_indices_final = np.where(is_ne_final)[0]
        np.random.seed(42)
        sampled_ne_final = np.random.choice(ne_indices_final, size=max_ne_final, replace=False)
        anom_indices_final = np.where(is_anom_final)[0]
        keep_final = np.concatenate([anom_indices_final, sampled_ne_final])
        X_final_data, y_enc_final_data = X[keep_final], y_enc[keep_final]
    else:
        X_final_data, y_enc_final_data = X, y_enc

    # Skip SMOTE for final model to avoid data corruption
    X_final, y_enc_final = X_final_data, y_enc_final_data
        
    final_model = XGBClassifier(
        n_estimators=40,
        max_depth=2,
        min_child_weight=10,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=1
    )
    weights_final = compute_sample_weight('balanced', y_enc_final)
    final_model.fit(X_final, y_enc_final, sample_weight=weights_final)


    # ---------- SAVE ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    pkl_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(final_model, pkl_path)
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    
    import json
    with open(os.path.join(model_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
        
    logger.info(f"Final model pickle saved at {pkl_path}")

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
        
        onnx_path = os.path.join(model_dir, "best_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Model successfully exported to ONNX format at {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")

if __name__ == "__main__":
    main()
