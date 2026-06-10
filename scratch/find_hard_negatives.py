import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("out/manual_labeled_windows.csv")

# Filter only originals

# pyrefly: ignore [missing-import]
from config import BEST_FEATURES
X = df[BEST_FEATURES].values
y = df["label"].values
groups = df["trip_id"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
p_idx = list(le.classes_).index("Pothole")
ne_idx = list(le.classes_).index("Non-Event")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))
oof_probs = np.zeros(len(df))

for train_idx, test_idx in cv.split(X, y_enc, groups):
    X_train, y_train = X[train_idx], y_enc[train_idx]
    
    # Undersample
    is_ne = (y_train == ne_idx)
    n_anom = np.sum(~is_ne)
    max_ne = int(n_anom * 1.5)
    ne_indices = np.where(is_ne)[0]
    if len(ne_indices) > max_ne:
        np.random.seed(42)
        sampled_ne = np.random.choice(ne_indices, size=max_ne, replace=False)
        anom_indices = np.where(~is_ne)[0]
        keep = np.concatenate([anom_indices, sampled_ne])
        X_train, y_train = X_train[keep], y_train[keep]
        
    model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced_subsample', random_state=42)
    model.fit(X_train, y_train)
    
    oof_probs[test_idx] = model.predict_proba(X[test_idx])[:, p_idx]
    oof_preds[test_idx] = model.predict(X[test_idx])

df["pothole_prob"] = oof_probs
df["pred_label"] = le.inverse_transform(oof_preds.astype(int))

fp_mask = (df["label"] == "Non-Event") & (df["pred_label"] == "Pothole")
fp_df = df[fp_mask].sort_values("pothole_prob", ascending=False)
tp_mask = (df["label"] == "Pothole") & (df["pred_label"] == "Pothole")
tp_df = df[tp_mask].sort_values("pothole_prob", ascending=False)

print(f"Total False Positives: {len(fp_df)}")
print(f"Total True Positives: {len(tp_df)}")

# Dump to CSV for inspection
fp_df.to_csv("scratch/false_positives.csv", index=False)
tp_df.to_csv("scratch/true_positives.csv", index=False)
print("Saved FP and TP to scratch/")
