import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from sklearn.isotonic import IsotonicRegression

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import InceptionTime1D

from data_utils import get_stratified_group_split

def evaluate_xgb():
    print("\n--- Evaluasi XGBoost ---")
    data_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])
    
    # Load feature columns
    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)
        
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values
    event_ids = df['event_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))
        
    _, test_groups_list = get_stratified_group_split(groups, y, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_mask]])
    X_test = X[test_mask][is_original_test]
    y_test_raw = y[test_mask][is_original_test]
    event_ids_test = event_ids[test_mask][is_original_test]
    
    # Load model
    model = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl"))
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    
    y_test = le.transform(y_test_raw)
    classes = le.classes_
    
    # Force default threshold evaluation (Argmax equivalent)
    best_thresh_p = 0.5
    best_thresh_sb = 0.5
    
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0
    
    raw_probas = model.predict_proba(X_test)
    
    # Apply isotonic calibration if calibrators exist
    cal_path = os.path.join(XGB_MODEL_DIR, "xgboost_calibrators.pkl")
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)
        probas = np.column_stack([
            calibrators[i].predict(raw_probas[:, i]) for i in range(len(classes))
        ])
        probas = probas / probas.sum(axis=1, keepdims=True)
        print("  Applied isotonic calibration to probabilities.")
    else:
        probas = raw_probas
    
    preds = np.zeros_like(y_test)
    for i in range(len(probas)):
        proba = probas[i]
        p_prob = proba[p_idx]
        sb_prob = proba[sb_idx] if sb_idx != -1 else 0.0
        
        p_triggered = p_prob >= best_thresh_p
        sb_triggered = sb_idx != -1 and sb_prob >= best_thresh_sb
        
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
            
    print(classification_report(y_test, preds, target_names=classes, zero_division=0))
    return event_ids_test, y_test, preds, classes

def evaluate_cnn():
    print("\n--- Evaluasi 1D-CNN ---")
    X_path = os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy")
    groups_path = os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy")
    event_ids_path = os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy")
    
    X_all = np.load(X_path)
    y_raw_all = np.load(y_path)
    groups_all = np.load(groups_path)
    event_ids_all = np.load(event_ids_path)
    
    _, test_groups_list = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.7)
    test_mask = np.isin(groups_all, test_groups_list)
    
    X_test_np = X_all[test_mask]
    y_test_raw = y_raw_all[test_mask]
    event_ids_test = event_ids_all[test_mask]
    
    # Load Label Encoder Classes
    classes = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_classes.npy"), allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    y_test = le.transform(y_test_raw)
    
    # Load Global Scaler
    scaler_path = os.path.join(CNN_MODEL_DIR, "cnn_1d_scaler_params.json")
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    global_means = np.array(scaler_params['means']).reshape(1, 7, 1)
    global_stds = np.array(scaler_params['stds']).reshape(1, 7, 1)
        
    X_test_scaled = (X_test_np - global_means) / global_stds
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # Load Model
    model = InceptionTime1D(in_channels=7, num_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probas = torch.sigmoid(outputs).numpy()
        
    preds = np.argmax(probas, axis=1)
            
    print(classification_report(y_test, preds, target_names=classes, zero_division=0))
    return event_ids_test, y_test, preds, classes

if __name__ == "__main__":
    event_ids_xgb, y_true_xgb, y_pred_xgb, classes_xgb = evaluate_xgb()
    event_ids_cnn, y_true_cnn, y_pred_cnn, classes_cnn = evaluate_cnn()
    
    # Save a comparison summary table on each model's holdout split directly.
    print("\n=== PERBANDINGAN PERFORMA HOLDOUT TEST SET ===")
    from sklearn.metrics import precision_recall_fscore_support
    
    metrics_xgb = precision_recall_fscore_support(y_true_xgb, y_pred_xgb, average=None, labels=range(len(classes_xgb)), zero_division=0)
    metrics_cnn = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, average=None, labels=range(len(classes_cnn)), zero_division=0)
    
    comparison_data = []
    for idx, cls in enumerate(classes_xgb):
        comparison_data.append({
            "Class": cls,
            "XGBoost Precision": metrics_xgb[0][idx],
            "1D-CNN Precision": metrics_cnn[0][idx],
            "XGBoost Recall": metrics_xgb[1][idx],
            "1D-CNN Recall": metrics_cnn[1][idx],
            "XGBoost F1-score": metrics_xgb[2][idx],
            "1D-CNN F1-score": metrics_cnn[2][idx],
        })
        
    df_comp = pd.DataFrame(comparison_data)
    print(df_comp.to_string(index=False))
    
    df_comp.to_csv(os.path.join(BASE_DIR, "evaluation", "reports", "comparison_report.csv"), index=False)
    print(f"\nLaporan perbandingan berhasil disimpan di: {os.path.join(BASE_DIR, 'evaluation', 'reports', 'comparison_report.csv')}")
