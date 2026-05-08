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
    # Columns to exclude from feature matrix
    exclude_cols = ['window_start', 'window_end', 'trip_id', 'event_id', 'source', 'label']
    
    # All other columns are considered features
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Drop rows with NaN in features
    df = df.dropna(subset=feature_cols)

    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values

    # Pastikan minimal ada 2 group (trip_id) agar GroupShuffleSplit berfungsi
    if len(np.unique(groups)) < 2:
        logger.warning("Hanya ada 1 trip_id di dalam dataset. Menggunakan fallback random train_test_split (risiko data leakage).")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # ---------- TRAIN/TEST SPLIT (GROUPED) ----------
        # We use GroupShuffleSplit to ensure windows from the same trip do not 
        # leak across the train and test sets.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Pastikan data test dan train valid
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Train atau Test set kosong setelah di split. Dataset terlalu kecil.")
        return

    # ---------- STRICT NORMALIZATION ----------
    # Fit the scaler ONLY on the training set to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- MODEL TRAINING ----------
    # Using balanced class weight to handle the Non-Event vs Event class imbalance.
    logger.info("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)

    # ---------- EVALUATION ----------
    logger.info("Evaluating model on test set...")
    y_pred = rf_model.predict(X_test_scaled)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))

    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    cm_df = pd.DataFrame(cm, index=rf_model.classes_, columns=rf_model.classes_)
    print(cm_df)

    # ---------- EXPORT ARTIFACTS ----------
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "rf_model.pkl")

    joblib.dump(scaler, scaler_path)
    joblib.dump(rf_model, model_path)

    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
