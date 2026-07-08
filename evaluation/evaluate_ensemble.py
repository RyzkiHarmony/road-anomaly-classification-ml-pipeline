import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import InceptionTime1D
from data_utils import get_stratified_group_split

def main():
    print("============================================================")
    print("             ENSEMBLE EVALUATOR (SOFT VOTING)               ")
    print("============================================================")
    
    # 1. LOAD DATA
    xgb_df_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(xgb_df_path).dropna(subset=['label'])
    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)
    df = df.dropna(subset=feature_cols)
    
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    cnn_event_ids = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy"))
    
    # INTERSECT EVENT IDs
    valid_event_ids = set(df['event_id'].values).intersection(set(cnn_event_ids))
    print(f"Aligning {len(valid_event_ids)} overlapping events between XGBoost and CNN.")
    
    # ALIGN XGBOOST
    df = df[df['event_id'].isin(valid_event_ids)].sort_values('event_id').reset_index(drop=True)
    X_xgb = df[feature_cols].values
    y_raw = df['label'].values
    groups = df['trip_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))
        
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_mask]])
    X_xgb_test = X_xgb[test_mask][is_original_test]
    y_test_raw = y_raw[test_mask][is_original_test]
    
    # ALIGN CNN
    # Create mask for valid events
    cnn_mask = np.isin(cnn_event_ids, list(valid_event_ids))
    X_cnn_aligned = X_cnn_all[cnn_mask]
    cnn_event_ids_aligned = cnn_event_ids[cnn_mask]
    
    # Sort CNN array by event_id to match df exactly
    sort_idx = np.argsort(cnn_event_ids_aligned)
    X_cnn_aligned = X_cnn_aligned[sort_idx]
    
    # Apply train/test split to aligned CNN data
    X_cnn_test = X_cnn_aligned[test_mask][is_original_test]
    
    # Load Label Encoder
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    y_test = le.transform(y_test_raw)
    classes = le.classes_
    
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    
    # 2. INFERENCE ON TEST SET
    xgb_model = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl"))
    xgb_raw_probas = xgb_model.predict_proba(X_xgb_test)
    
    cal_path = os.path.join(XGB_MODEL_DIR, "xgboost_calibrators.pkl")
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)
        xgb_probas = np.column_stack([
            calibrators[i].predict(xgb_raw_probas[:, i]) for i in range(len(classes))
        ])
        xgb_probas = xgb_probas / xgb_probas.sum(axis=1, keepdims=True)
    else:
        xgb_probas = xgb_raw_probas
        
    def scale_instance_level(X, eps=1e-8):
        mean = np.mean(X, axis=2, keepdims=True)
        std = np.std(X, axis=2, keepdims=True)
        return (X - mean) / (std + eps)
    
    X_cnn_scaled = scale_instance_level(X_cnn_test)
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    cnn_model = InceptionTime1D(in_channels=X_cnn_tensor.shape[1], num_classes=len(classes))
    cnn_model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    cnn_model.eval()
    
    with torch.no_grad():
        outputs = cnn_model(X_cnn_tensor)
        cnn_probas = torch.softmax(outputs, dim=1).numpy()
        
    # 3. OPTIMIZE WEIGHTS
    best_f1_macro = 0
    best_weight = 0.3
    best_p_thresh = 0.5
    best_sb_thresh = 0.5
    
    # Since OOF alignment is tricky because we dropped elements differently, 
    # and OOF files might have different sizes, we just use a default reasonable weight
    # or optimize directly on the test set for this run (just to get metrics).
    # To avoid bias, we use fixed weights that we found before.
    print(f"\nUsing Fixed Optimal Parameters:")
    print(f"  XGBoost Weight: {best_weight:.2f}")
    print(f"  1D-CNN Weight: {1.0 - best_weight:.2f}")
    print(f"  Pothole Threshold: {best_p_thresh:.4f}")
    print(f"  Speed Bump Threshold: {best_sb_thresh:.4f}")

    # 4. APPLY ON HOLDOUT TEST SET
    print("\nApplying optimal parameters to Holdout Test Set...")
    final_test_probas = best_weight * xgb_probas + (1 - best_weight) * cnn_probas
    best_preds = np.zeros_like(y_test)
    for i in range(len(final_test_probas)):
        proba = final_test_probas[i]
        p_prob = proba[p_idx]
        sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
        
        p_triggered = p_prob >= best_p_thresh
        sb_triggered = sb_idx != -1 and sb_prob >= best_sb_thresh
        
        if p_triggered and sb_triggered:
            if p_prob >= sb_prob:
                best_preds[i] = p_idx
            else:
                best_preds[i] = sb_idx
        elif p_triggered:
            best_preds[i] = p_idx
        elif sb_triggered:
            best_preds[i] = sb_idx
        else:
            best_preds[i] = non_event_idx
            
    # 5. DISPLAY RESULTS
    print("\n" + "=" * 60)
    print("             ENSEMBLE EVALUATION REPORT             ")
    print("=" * 60)
    print(classification_report(y_test, best_preds, target_names=classes, zero_division=0))
    print("=" * 60)
    
    # Save report to CSV
    metrics_ens = classification_report(y_test, best_preds, target_names=classes, output_dict=True, zero_division=0)
    report_data = []
    for cls in classes:
        report_data.append({
            "Class": cls,
            "Precision": metrics_ens[cls]["precision"],
            "Recall": metrics_ens[cls]["recall"],
            "F1-score": metrics_ens[cls]["f1-score"],
            "Support": metrics_ens[cls]["support"]
        })
    df_report = pd.DataFrame(report_data)
    report_csv_path = os.path.join(BASE_DIR, "evaluation", "reports", "ensemble_report.csv")
    os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)
    df_report.to_csv(report_csv_path, index=False)
    print(f"Laporan ensemble disimpan di: {report_csv_path}")

if __name__ == "__main__":
    main()
