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
    
    # ---------- FEATURE SELECTION ----------
    # Kita gunakan fitur yang sama dengan label_suggester.py untuk konsistensi
    from label_suggester import ML_FEATURES
    
    # Tambahkan fitur tambahan jika tersedia di dataset baru
    feature_cols = [c for c in df.columns if c in ML_FEATURES or c in [
        "asymmetry_score", "kurtosis", "skewness", "fft_high_low_ratio", 
        "zcr", "gyro_pitch_roll_ratio", "peak_to_peak"
    ]]
    
    # Drop rows with NaN in features
    df = df.dropna(subset=feature_cols)

    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values

    # ---------- TRAIN/TEST SPLIT (GROUPED) ----------
    if len(np.unique(groups)) < 2:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # ---------- MODEL ARENA (RF vs XGB vs LGBM) ----------
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import LabelEncoder, label_binarize
    
    # Label Encoding for multi-class
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', max_depth=10, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', eval_metric='mlogloss', n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1,
            class_weight='balanced', n_jobs=-1, verbose=-1
        )
    }

    best_model_name = None
    best_f1 = -1
    best_model_obj = None

    print("\n" + "="*50)
    print("ALGORITHM COMPARISON ARENA")
    print("="*50)

    # Prepare binarized labels for PR-AUC
    y_test_bin = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # XGBoost handles weights differently
        if name == "XGBoost":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_enc)
            model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
            y_pred_proba = model.predict_proba(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        print(f"\n--- {name} ---")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Calculate PR-AUC for Pothole
        try:
            p_idx = list(classes).index("Pothole")
            prauc = average_precision_score(y_test_bin[:, p_idx], y_pred_proba[:, p_idx])
            print(f"Pothole PR-AUC: {prauc:.3f}")
        except Exception:
            prauc = 0

        # Optimization Target: Pothole F1
        p_f1 = report.get('Pothole', {}).get('f1-score', 0)
        if p_f1 > best_f1:
            best_f1 = p_f1
            best_model_name = name
            best_model_obj = model

    print("="*50)
    logger.info(f"🏆 Best Model for Potholes: {best_model_name} (F1: {best_f1:.2f})")

    # ---------- EXPORT ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model_obj, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    logger.info(f"Best model ({best_model_name}) and encoder saved to {model_dir}")

if __name__ == "__main__":
    main()
