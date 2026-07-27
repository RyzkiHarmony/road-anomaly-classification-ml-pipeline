import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import InceptionTime1D
from data_utils import get_stratified_group_split

def main():
    print("Loading CNN data for comparison visualization...")
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    y_raw = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"))
    groups = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"))
    
    # Stratified split to match the holdout test set in train.py
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    X_cnn_test = X_cnn_all[test_mask]
    y_test_raw = y_raw[test_mask]
    groups_test = groups[test_mask]
    
    df_test = pd.DataFrame({
        'label': y_test_raw,
        'trip_id': groups_test
    })
    
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    classes = le.classes_
    scaler_path = os.path.join(CNN_MODEL_DIR, "cnn_1d_scaler_params.json")
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    global_means = np.array(scaler_params['means']).reshape(1, 7, 1)
    global_stds = np.array(scaler_params['stds']).reshape(1, 7, 1)
    
    X_cnn_scaled = (X_cnn_test - global_means) / global_stds
    seq_len = 200
    if X_cnn_scaled.shape[-1] > seq_len:
        start_idx = (X_cnn_scaled.shape[-1] - seq_len) // 2
        X_cnn_scaled = X_cnn_scaled[..., start_idx:start_idx + seq_len]
        X_cnn_test = X_cnn_test[..., start_idx:start_idx + seq_len]
        
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    cnn_model = InceptionTime1D(in_channels=X_cnn_tensor.shape[1], num_classes=len(classes), channels=64, bottleneck_channels=16)
    cnn_model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    cnn_model.eval()
    
    with torch.no_grad():
        outputs = cnn_model(X_cnn_tensor)
        cnn_probas = torch.softmax(outputs, dim=1).numpy()
        
    cnn_preds = np.argmax(cnn_probas, axis=1)
    df_test['cnn_prediction'] = le.inverse_transform(cnn_preds)
    df_test['cnn_confidence_pred'] = np.max(cnn_probas, axis=1)
    
    if "Pothole" in list(classes):
        p_idx = list(classes).index("Pothole")
        df_test['prob_Pothole'] = cnn_probas[:, p_idx]
    
    # Filter FN: Pothole -> Non-Event
    fn_mask = (df_test['label'] == 'Pothole') & (df_test['cnn_prediction'] == 'Non-Event')
    # Filter TP: Pothole -> Pothole
    tp_mask = (df_test['label'] == 'Pothole') & (df_test['cnn_prediction'] == 'Pothole')
    
    df_fn = df_test[fn_mask].copy()
    X_fn = X_cnn_test[fn_mask]
    
    df_tp = df_test[tp_mask].copy()
    X_tp = X_cnn_test[tp_mask]
    
    # Select best representations
    # High confidence TP
    tp_idx = np.argmax(df_tp['cnn_confidence_pred'].values)
    tp_row = df_tp.iloc[tp_idx]
    tp_sig = X_tp[tp_idx]
    
    # High confidence FN (model is very sure it's non-event)
    fn_idx = np.argmax(df_fn['cnn_confidence_pred'].values)
    fn_row = df_fn.iloc[fn_idx]
    fn_sig = X_fn[fn_idx]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.3, top=0.9)
    
    fig.suptitle("Analisis Error: Perbandingan Sinyal Pothole (True Positive vs False Negative)", fontsize=16, fontweight='bold')
    
    t = np.linspace(0, 2.0, 200)
    
    # --- TRUE POSITIVE (LEFT COLUMN) ---
    axes[0,0].plot(t, tp_sig[1], label='Acc X', linewidth=1.5)
    axes[0,0].plot(t, tp_sig[2], label='Acc Y', linewidth=1.5)
    axes[0,0].plot(t, tp_sig[3], label='Acc Z', linewidth=1.5)
    axes[0,0].set_title(f'True Positive (Pothole)\nProb: {tp_row.get("prob_Pothole",0):.3f} | Speed: {tp_sig[0].mean():.1f} m/s')
    axes[0,0].legend(loc='upper right')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[1,0].plot(t, tp_sig[0], label='Speed (m/s)', color='magenta', linewidth=1.5)
    axes[1,0].set_ylabel('Speed')
    axes[1,0].legend(loc='upper right')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[2,0].plot(t, tp_sig[4], label='Gyro X', linewidth=1.5)
    axes[2,0].plot(t, tp_sig[5], label='Gyro Y', linewidth=1.5)
    axes[2,0].plot(t, tp_sig[6], label='Gyro Z', linewidth=1.5)
    axes[2,0].set_xlabel('Time (seconds)')
    axes[2,0].legend(loc='upper right')
    axes[2,0].grid(True, alpha=0.3)
    
    # --- FALSE NEGATIVE (RIGHT COLUMN) ---
    axes[0,1].plot(t, fn_sig[1], label='Acc X', linewidth=1.5)
    axes[0,1].plot(t, fn_sig[2], label='Acc Y', linewidth=1.5)
    axes[0,1].plot(t, fn_sig[3], label='Acc Z', linewidth=1.5)
    axes[0,1].set_title(f'False Negative (Missed Pothole)\nPred: Non-Event ({fn_row["cnn_confidence_pred"]:.3f}) | Speed: {fn_sig[0].mean():.1f} m/s')
    axes[0,1].legend(loc='upper right')
    axes[0,1].grid(True, alpha=0.3)
    
    # Set y limits to match left column for fair visual comparison
    axes[0,1].set_ylim(axes[0,0].get_ylim())
    axes[1,1].set_ylim(axes[1,0].get_ylim())
    axes[2,1].set_ylim(axes[2,0].get_ylim())

    axes[1,1].plot(t, fn_sig[0], label='Speed (m/s)', color='magenta', linewidth=1.5)
    axes[1,1].legend(loc='upper right')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[2,1].plot(t, fn_sig[4], label='Gyro X', linewidth=1.5)
    axes[2,1].plot(t, fn_sig[5], label='Gyro Y', linewidth=1.5)
    axes[2,1].plot(t, fn_sig[6], label='Gyro Z', linewidth=1.5)
    axes[2,1].set_xlabel('Time (seconds)')
    axes[2,1].legend(loc='upper right')
    axes[2,1].grid(True, alpha=0.3)

    os.makedirs(os.path.join(BASE_DIR, "artifacts"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "artifacts", "pothole_error_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
