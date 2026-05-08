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

    # ---------- MODEL TRAINING (NO SCALER) ----------
    # RF tidak butuh scaling. Ini membuat model lebih "physically grounded".
    logger.info("Training RandomForestClassifier (Scale-Invariant)...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)

    # ---------- EVALUATION ----------
    logger.info("Evaluating model on test set...")
    y_pred = rf_model.predict(X_test)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))

    # ---------- EXPORT ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "rf_model.pkl")
    joblib.dump(rf_model, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
