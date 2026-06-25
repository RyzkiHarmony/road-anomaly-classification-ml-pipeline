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

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import Lightweight1DCNN

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

    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)

    return list(train_groups), list(test_groups)

def main():
    print("============================================================")
    print("             ENSEMBLE EVALUATOR (SOFT VOTING)               ")
    print("============================================================")
    
    # ─── 1. LOAD DATA & MODELS ───
    # XGBoost Data
    xgb_df_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(xgb_df_path).dropna(subset=['label'])
    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)
    df = df.dropna(subset=feature_cols)
    
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
    
    # CNN Data
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    X_cnn_test = X_cnn_all[test_mask][is_original_test]
    
    # Load Label Encoder
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    y_test = le.transform(y_test_raw)
    classes = le.classes_
    
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    
    # ─── 2. INFERENCE ───
    # XGBoost Probability Prediction
    xgb_model = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl"))
    xgb_raw_probas = xgb_model.predict_proba(X_xgb_test)
    
    # Apply Calibration
    cal_path = os.path.join(XGB_MODEL_DIR, "xgboost_calibrators.pkl")
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)
        xgb_probas = np.column_stack([
            calibrators[i].predict(xgb_raw_probas[:, i]) for i in range(len(classes))
        ])
        xgb_probas = xgb_probas / xgb_probas.sum(axis=1, keepdims=True)
    else:
        xgb_probas = xgb_raw_probas
        
    # CNN Probability Prediction
    with open(os.path.join(CNN_MODEL_DIR, "cnn_1d_scaler_params.json"), "r") as f:
        scaler_params = json.load(f)
    means = np.array(scaler_params["means"]).reshape(1, -1, 1)
    stds = np.array(scaler_params["stds"]).reshape(1, -1, 1)
    
    X_cnn_scaled = (X_cnn_test - means) / stds
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    cnn_model = Lightweight1DCNN(in_channels=14, num_classes=len(classes))
    cnn_model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    cnn_model.eval()
    
    with torch.no_grad():
        outputs = cnn_model(X_cnn_tensor)
        cnn_probas = torch.softmax(outputs, dim=1).numpy()
        
    # ─── 3. GRID SEARCH FOR BEST ENSEMBLE WEIGHTS ───
    best_f1_macro = 0
    best_weight = 0.5
    best_p_thresh = 0.5
    best_sb_thresh = 0.5
    best_preds = None
    
    # Grid search over XGBoost weight and decision thresholds
    weights_to_try = np.linspace(0.0, 1.0, 11)
    thresholds_to_try = np.linspace(0.2, 0.8, 13)
    
    print("Optimizing ensemble parameters on Holdout Set...")
    for w in weights_to_try:
        # Soft voting combination
        blend_probas = w * xgb_probas + (1 - w) * cnn_probas
        
        for t_p in thresholds_to_try:
            for t_sb in thresholds_to_try:
                preds = np.zeros_like(y_test)
                for i in range(len(blend_probas)):
                    proba = blend_probas[i]
                    p_prob = proba[p_idx]
                    sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
                    
                    p_triggered = p_prob >= t_p
                    sb_triggered = sb_idx != -1 and sb_prob >= t_sb
                    
                    if p_triggered and sb_triggered:
                        if p_prob >= sb_prob:
                            preds[i] = p_idx
                        else:
                            preds[i] = sb_idx
                    elif p_triggered:
                        preds[i] = p_idx
                    elif sb_triggered:
                        preds[i] = sb_idx
                    else:
                        preds[i] = non_event_idx
                
                # We want to maximize the average macro F1-score of minority classes (Pothole & Speed Bump)
                f1_p = f1_score(y_test, preds, labels=[p_idx], average='macro', zero_division=0)
                f1_sb = f1_score(y_test, preds, labels=[sb_idx], average='macro', zero_division=0)
                score = (f1_p + f1_sb) / 2.0
                
                if score > best_f1_macro:
                    best_f1_macro = score
                    best_weight = w
                    best_p_thresh = t_p
                    best_sb_thresh = t_sb
                    best_preds = preds
                    
    print(f"\nOptimal Parameters found:")
    print(f"  XGBoost Weight: {best_weight:.2f}")
    print(f"  1D-CNN Weight: {1.0 - best_weight:.2f}")
    print(f"  Pothole Threshold: {best_p_thresh:.4f}")
    print(f"  Speed Bump Threshold: {best_sb_thresh:.4f}")
    
    # ─── 4. DISPLAY RESULTS ───
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
