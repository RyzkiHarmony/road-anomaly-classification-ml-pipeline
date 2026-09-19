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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))
sys.path.append(os.path.join(BASE_DIR, "src", "stage3_eda_and_splitting"))
sys.path.append(os.path.join(BASE_DIR, "src", "stage4_modeling", "cnn"))

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import InceptionTime1D
from data_splitting import get_stratified_group_split

def evaluate_xgb():
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
        
    _, test_groups_list = get_stratified_group_split(groups, y, train_ratio=0.8)
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
    
    raw_probas = model.predict_proba(X_test)
    
    # Apply isotonic calibration if calibrators exist
    cal_path = os.path.join(XGB_MODEL_DIR, "xgboost_calibrators.pkl")
    if os.path.exists(cal_path):
        calibrators = joblib.load(cal_path)
        probas = np.column_stack([
            calibrators[i].predict(raw_probas[:, i]) for i in range(len(classes))
        ])
        probas = probas / probas.sum(axis=1, keepdims=True)
        cal_status = "Isotonic Regression (Aktif)"
    else:
        probas = raw_probas
        cal_status = "Tanpa Kalibrasi (Raw Softmax)"
    
    preds = np.argmax(probas, axis=1)

    print("\n[1/3] EVALUASI MODEL TABULAR (XGBOOST + OPTUNA TUNED):", flush=True)
    print(f"  - Fitur Digunakan       : {len(feature_cols)} fitur non-redundan terpilih", flush=True)
    print(f"  - Kalibrasi Probabilitas: {cal_status}", flush=True)
    print(f"  - Sampel Holdout Test   : {len(y_test):,} sampel ({len(test_groups_list)} rute baru - murni data asli)", flush=True)
    print("  " + "-" * 70, flush=True)
    print(classification_report(y_test, preds, target_names=classes, digits=4, zero_division=0), flush=True)

    return event_ids_test, y_test, preds, classes


def evaluate_cnn():
    X_path = os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy")
    groups_path = os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy")
    event_ids_path = os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy")
    
    X_all = np.load(X_path)
    y_raw_all = np.load(y_path)
    groups_all = np.load(groups_path)
    event_ids_all = np.load(event_ids_path)
    
    _, test_groups_list = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.8)
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
    
    # Load best params to match saved checkpoint
    best_params_path = os.path.join(BASE_DIR, "src", "stage4_modeling", "cnn", "best_optuna_params.json")
    
    channels_val = 32
    dropout_val = 0.5
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            bp = json.load(f)
            channels_val = bp.get('channels', channels_val)
            dropout_val = bp.get('dropout', dropout_val)

    # Load Model
    model = InceptionTime1D(
        in_channels=7,
        num_classes=len(classes),
        num_blocks=3,
        channels=channels_val,
        bottleneck_channels=channels_val // 4,
        dropout_rate=dropout_val
    )
    model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probas = torch.softmax(outputs, dim=1).numpy()
        
    preds = np.argmax(probas, axis=1)

    print("\n[2/3] EVALUASI MODEL DEEP LEARNING (1D-CNN INCEPTIONTIME):", flush=True)
    print(f"  - Arsitektur Jaringan   : InceptionTime1D (7 Channels, 3 Blocks, Channels={channels_val})", flush=True)
    print("  - Normalisasi Input     : Global Mean & Std Z-Score Scaler", flush=True)
    print(f"  - Sampel Holdout Test   : {len(y_test):,} sampel ({len(test_groups_list)} rute baru - murni data asli)", flush=True)
    print("  " + "-" * 70, flush=True)
    print(classification_report(y_test, preds, target_names=classes, digits=4, zero_division=0), flush=True)

    return event_ids_test, y_test, preds, classes


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    print("\n" + "=" * 82, flush=True)
    print("     EVALUASI KOMPARATIF & STUDI ABLASI MODEL (HOLDOUT TEST SET 80:20)", flush=True)
    print("=" * 82, flush=True)

    event_ids_xgb, y_true_xgb, y_pred_xgb, classes_xgb = evaluate_xgb()
    event_ids_cnn, y_true_cnn, y_pred_cnn, classes_cnn = evaluate_cnn()
    
    metrics_xgb = precision_recall_fscore_support(y_true_xgb, y_pred_xgb, average=None, labels=range(len(classes_xgb)), zero_division=0)
    metrics_cnn = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, average=None, labels=range(len(classes_cnn)), zero_division=0)

    acc_xgb = accuracy_score(y_true_xgb, y_pred_xgb)
    acc_cnn = accuracy_score(y_true_cnn, y_pred_cnn)

    macro_xgb = precision_recall_fscore_support(y_true_xgb, y_pred_xgb, average='macro', zero_division=0)
    macro_cnn = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, average='macro', zero_division=0)

    print("\n" + "=" * 82, flush=True)
    print("  [3/3] STUDI ABLASI: PERBANDINGAN PERFORMA HEAD-TO-HEAD (XGBOOST vs 1D-CNN)", flush=True)
    print("=" * 82, flush=True)
    div_line = "  " + "-" * 78
    print(div_line, flush=True)
    print(f"  {'Kelas / Evaluasi':<22} {'Model':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}", flush=True)
    print(div_line, flush=True)

    comparison_rows = []

    for idx, cls in enumerate(classes_xgb):
        supp_xgb = int(np.sum(y_true_xgb == idx))
        supp_cnn = int(np.sum(y_true_cnn == idx))
        
        print(f"  {cls:<22} {'XGBoost':<12} {metrics_xgb[0][idx]:>10.4f} {metrics_xgb[1][idx]:>10.4f} {metrics_xgb[2][idx]:>10.4f} {supp_xgb:>10}", flush=True)
        print(f"  {'':<22} {'1D-CNN':<12} {metrics_cnn[0][idx]:>10.4f} {metrics_cnn[1][idx]:>10.4f} {metrics_cnn[2][idx]:>10.4f} {supp_cnn:>10}", flush=True)
        print(div_line, flush=True)

        comparison_rows.append({
            "Class": cls,
            "XGBoost Precision": round(float(metrics_xgb[0][idx]), 4),
            "1D-CNN Precision": round(float(metrics_cnn[0][idx]), 4),
            "XGBoost Recall": round(float(metrics_xgb[1][idx]), 4),
            "1D-CNN Recall": round(float(metrics_cnn[1][idx]), 4),
            "XGBoost F1-score": round(float(metrics_xgb[2][idx]), 4),
            "1D-CNN F1-score": round(float(metrics_cnn[2][idx]), 4),
            "XGBoost Support": supp_xgb,
            "1D-CNN Support": supp_cnn,
        })

    print(f"  {'Macro Average':<22} {'XGBoost':<12} {macro_xgb[0]:>10.4f} {macro_xgb[1]:>10.4f} {macro_xgb[2]:>10.4f} {len(y_true_xgb):>10}", flush=True)
    print(f"  {'':<22} {'1D-CNN':<12} {macro_cnn[0]:>10.4f} {macro_cnn[1]:>10.4f} {macro_cnn[2]:>10.4f} {len(y_true_cnn):>10}", flush=True)
    print(div_line, flush=True)

    print(f"  {'Overall Accuracy':<22} {'XGBoost':<12} {acc_xgb:>10.4f} ({acc_xgb*100:.2f}%)", flush=True)
    print(f"  {'':<22} {'1D-CNN':<12} {acc_cnn:>10.4f} ({acc_cnn*100:.2f}%)", flush=True)
    print(div_line, flush=True)

    # Save comparison dataframe
    df_comp = pd.DataFrame(comparison_rows)
    report_path = os.path.join(BASE_DIR, "evaluation", "reports", "comparison_report.csv")
    df_comp.to_csv(report_path, index=False)

    rel_report = os.path.relpath(report_path, BASE_DIR).replace('\\', '/')
    print("\n[STATUS PENYIMPANAN LAPORAN]", flush=True)
    print(f"  - File CSV : {rel_report}", flush=True)
    print("=" * 82 + "\n", flush=True)
