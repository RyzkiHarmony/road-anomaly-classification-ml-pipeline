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
    
    # ---------- FEATURE SELECTION (TOP 14 ONLY) ----------
    # Berdasarkan audit Senior ML Engineer, kita pangkas fitur noise.
    # BEST_FEATURES diimport dari config.py agar tersentralisasi sebagai SSOT
    
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)

    logger.info(f"Using TOP {len(feature_cols)} features to prevent overfitting: {feature_cols}")

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

        # Handle Class Imbalance using SMOTE in Feature Space
        from imblearn.over_sampling import SMOTE
        # Only apply SMOTE if there are enough samples in minority class (k_neighbors=5 by default)
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
            if min(np.bincount(y_train)) > 1: # ensure at least 2 samples for SMOTE
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            logger.warning(f"SMOTE failed on fold {fold+1}: {e}")

        # XGBoost is too greedy for 53 samples. Switching to Random Forest for better stability on tiny data.
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions on the clean, un-augmented test split
        y_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Track OOF
        oof_y_true.extend(y_test)
        oof_y_pred.extend(y_pred)
        if len(y_proba) > 0:
            oof_y_proba.append(y_proba)
        
        # Metric: Pothole F1
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        p_f1 = report.get(str(p_idx), {}).get('f1-score', 0)
        fold_metrics.append(p_f1)
        
        print(f"Fold {fold+1} | Pothole F1 (Unbiased): {p_f1:.4f} | Samples: {len(X_test)} (All Original)")

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
    
    # Cari threshold untuk target Recall > 0.65
    target_recall = 0.65
    idx = np.where(recalls >= target_recall)[0][-1] if len(np.where(recalls >= target_recall)[0]) > 0 else 0
    opt_threshold = thresholds[idx] if len(thresholds) > idx else 0.5
    
    print(f"Proposed Threshold for Pothole: {opt_threshold:.4f}")
    if len(precisions) > idx:
        print(f"Expected -> Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")

    # ---------- FINAL REPORT (CROSS-VALIDATION OOF) ----------
    print("\nFinal Report (Out-of-Fold - Unbiased Original):")
    print(classification_report(oof_y_true, oof_y_pred, target_names=classes, zero_division=0))

    # ---------- TRAIN FINAL PRODUCTION MODEL ----------
    logger.info("Training final production model on the entire dataset (Original + Augmented)...")
    try:
        from imblearn.over_sampling import SMOTE
        smote_final = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_enc)) - 1))
        X_final, y_enc_final = smote_final.fit_resample(X, y_enc)
    except Exception as e:
        logger.warning(f"SMOTE failed on final model: {e}")
        X_final, y_enc_final = X, y_enc
        
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_final, y_enc_final)

    # ---------- SAVE ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    pkl_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(final_model, pkl_path)
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    logger.info(f"Final model pickle saved at {pkl_path}")

    # ---------- ONNX EXPORT ----------
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        logger.info("Exporting final model to ONNX format for Android deployment...")
        initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
        
        # Convert the RandomForest final model to ONNX format
        onnx_model = convert_sklearn(
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
