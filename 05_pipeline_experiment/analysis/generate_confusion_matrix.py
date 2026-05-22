import sys
import os
# Ensure parent directory is in path for modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_SCRIPT_DIR))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

# Adjust paths to use absolute locations relative to this script
DATA_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'manual_labeled_windows.csv'))
MODEL_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'models', 'best_model.pkl'))
ENCODER_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'models', 'label_encoder.pkl'))
OUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'plots'))

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    print(f"Model or Dataset not found. Check:\n{MODEL_PATH}\n{DATA_PATH}")
    sys.exit(1)

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Re-create the Test Split using same BEST_FEATURES as training
BEST_FEATURES = [
    "event_duration", "speed_normalized_p2p", "peak_interval_std", 
    "vert_jrk", "kurtosis", "peak_mag", "peak_interval_mean", 
    "skewness", "gyro_roll_energy", "num_peaks_accel"
]
feature_cols = [c for c in df.columns if c in BEST_FEATURES]
df = df.dropna(subset=feature_cols + ['label'])

X = df[feature_cols].values
y = df['label'].values
groups = df['trip_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))

# Crucial ML fix: Evaluate ONLY on original (non-augmented) test data
# to measure true real-world generalization performance.
test_df = df.iloc[test_idx]
original_mask = test_df['source'].isna() | (~test_df['source'].str.contains('augmented', na=True))
test_df_original = test_df[original_mask]

X_test, y_test = test_df_original[feature_cols].values, test_df_original['label'].values

# Predict
y_pred = model.predict(X_test)
if isinstance(y_pred[0], (np.int32, np.int64, int)):
    y_pred = le.inverse_transform(y_pred)

# --- Confusion Matrix ---
classes = le.classes_
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix: Road Anomaly Detection (XGBoost)', fontsize=15, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
plot_path = os.path.join(OUT_DIR, 'confusion_matrix.png')
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")

# Print report for confirmation
print("\nFinal Test Evaluation:")
print(classification_report(y_test, y_pred, zero_division=0))
