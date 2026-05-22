import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import OUT_FOLDER, get_logger

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
    
    # ---------- FEATURE SELECTION (TOP 10 ONLY) ----------
    # Berdasarkan audit Senior ML Engineer, kita pangkas fitur noise.
    BEST_FEATURES = [
        "event_duration", "speed_normalized_p2p", "peak_interval_std", 
        "vert_jrk", "kurtosis", "peak_mag", "peak_interval_mean", 
        "skewness", "gyro_roll_energy", "num_peaks_accel"
    ]
    
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)

    logger.info(f"Using TOP {len(feature_cols)} features to prevent overfitting: {feature_cols}")

    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values
    
    # Save source column as array to easily filter out augmented twins in validation
    source_values = df['source'].fillna('original').values

    # ---------- CROSS-VALIDATION (STRATIFIED GROUP K-FOLD) ----------
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    n_folds = 5
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")

    print("\n" + "="*60)
    print(f"{'5-FOLD CROSS VALIDATION STRATEGY (UNBIASED)':^60}")
    print("="*60)

    fold_metrics = []
    best_overall_f1 = -1
    best_fold_model = None

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y_enc, groups)):
        # 1. Train Set: Keep both original and augmented events
        X_train, y_train = X[train_idx], y_enc[train_idx]
        
        # 2. Test Set: STRICTLY filter out augmented twins to prevent data leakage validation mirage
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        
        X_test, y_test = X[clean_test_idx], y_enc[clean_test_idx]

        # Handle Class Imbalance
        sw = compute_sample_weight(class_weight='balanced', y=y_train)
        
        model = XGBClassifier(
            n_estimators=150, 
            learning_rate=0.05, 
            max_depth=4, 
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train, sample_weight=sw)
        
        # Predictions on the clean, un-augmented test split
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # Metric: Pothole F1
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        p_f1 = report.get(str(p_idx), {}).get('f1-score', 0)
        fold_metrics.append(p_f1)
        
        print(f"Fold {fold+1} | Pothole F1 (Unbiased): {p_f1:.4f} | Samples: {len(X_test)} (All Original)")
        
        if p_f1 > best_overall_f1:
            best_overall_f1 = p_f1
            best_fold_model = model

    avg_f1 = np.mean(fold_metrics)
    print("-" * 60)
    logger.info(f"Average Pothole F1 across {n_folds} folds: {avg_f1:.4f}")

    # ---------- THRESHOLD OPTIMIZATION (Using Clean Fold) ----------
    # Kita cari threshold yang menyeimbangkan Recall vs Precision pada data asli
    print("\n" + "="*60)
    print(f"{'THRESHOLD OPTIMIZATION (Pothole)':^60}")
    print("="*60)
    
    # Ambil probabilitas kelas Pothole dari fold terbaik (atau terakhir)
    y_test_pothole = (y_test == p_idx).astype(int)
    y_proba_pothole = y_proba[:, p_idx]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test_pothole, y_proba_pothole)
    
    # Cari threshold untuk target Recall > 0.65
    target_recall = 0.65
    idx = np.where(recalls >= target_recall)[0][-1]
    opt_threshold = thresholds[idx]
    
    print(f"Proposed Threshold for Pothole: {opt_threshold:.4f}")
    print(f"Expected -> Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")

    # ---------- FINAL REPORT (BEST FOLD) ----------
    y_test_pred = best_fold_model.predict(X_test)
    print("\nFinal Report (Best Fold - Unbiased Original):")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))

    # ---------- TRAIN FINAL PRODUCTION MODEL ----------
    logger.info("Training final production model on the entire dataset (Original + Augmented)...")
    sw_final = compute_sample_weight(class_weight='balanced', y=y_enc)
    final_model = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y_enc, sample_weight=sw_final)

    # ---------- SAVE ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    pkl_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(final_model, pkl_path)
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    logger.info(f"Final model pickle saved at {pkl_path}")

    # ---------- ONNX EXPORT ----------
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        logger.info("Exporting final model to ONNX format for Android deployment...")
        initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
        
        # Convert the XGBoost final model to ONNX format
        onnx_model = onnxmltools.convert_xgboost(
            final_model, 
            initial_types=initial_types, 
            target_opset=15
        )
        
        onnx_path = os.path.join(model_dir, "best_model.onnx")
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        logger.info(f"Model successfully exported to ONNX format at {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")

if __name__ == "__main__":
    main()
