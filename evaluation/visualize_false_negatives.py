import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

from model import InceptionTime1D
from data_utils import get_stratified_group_split

def main():
    print("Loading CNN data for False Negative visualization...")
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    y_raw = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"))
    groups = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"))
    
    # Stratified split to match the holdout test set in train.py
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    X_cnn_test = X_cnn_all[test_mask]
    y_test_raw = y_raw[test_mask]
    groups_test = groups[test_mask]
    
    # Create DataFrame to track metadata
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
    # Apply center crop to 200 length like in evaluation
    seq_len = 200
    if X_cnn_scaled.shape[-1] > seq_len:
        start_idx = (X_cnn_scaled.shape[-1] - seq_len) // 2
        X_cnn_scaled = X_cnn_scaled[..., start_idx:start_idx + seq_len]
        X_cnn_test = X_cnn_test[..., start_idx:start_idx + seq_len]
        
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    cnn_model = InceptionTime1D(in_channels=X_cnn_tensor.shape[1], num_classes=len(classes))
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
    
    # Map trip_id to filename
    meta_dir = os.path.join(BASE_DIR, "data", "raw", "active", "meta")
    trip_to_filename = {}
    for meta_file in glob.glob(os.path.join(meta_dir, "*.json")):
        try:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
                if 'tripId' in meta_data:
                    base_name = os.path.basename(meta_file).replace('.json', '')
                    trip_to_filename[meta_data['tripId']] = base_name
        except Exception as e:
            pass
            
    unique_trips = df_test['trip_id'].unique()
    df_test['trip_index'] = df_test['trip_id'].apply(lambda x: np.where(unique_trips == x)[0][0] + 1)
    df_test['trip_filename'] = df_test['trip_id'].map(lambda x: trip_to_filename.get(x, 'Unknown Trip'))
    
    # Filter FN: Asli Pothole/Speed Bump tapi diprediksi Non-Event
    fn_mask = (df_test['label'].isin(['Pothole', 'Speed Bump'])) & (df_test['cnn_prediction'] == 'Non-Event')
    
    df_fn = df_test[fn_mask].copy()
    X_fn = X_cnn_test[fn_mask] # Raw unscaled data
    
    # Sort by probability of Non-Event
    sort_idx = np.argsort(-df_fn['cnn_confidence_pred'].values)
    df_fn = df_fn.iloc[sort_idx].reset_index(drop=True)
    X_fn = X_fn[sort_idx]
    
    if len(df_fn) == 0:
        print("No False Negatives found!")
        return

    print(f"Found {len(df_fn)} False Negatives. Launching interactive visualizer...")
    
    current_idx = [0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    def draw_plot():
        for ax in axes:
            ax.clear()
            
        idx = current_idx[0]
        row = df_fn.iloc[idx]
        sig = X_fn[idx] 
        
        sig_len = sig.shape[1]
        t = np.linspace(0, sig_len / 100.0, sig_len) 
        
        # Plot Raw Accelerometer (Indices 1,2,3)
        axes[0].plot(t, sig[1], label='Acc X')
        axes[0].plot(t, sig[2], label='Acc Y')
        axes[0].plot(t, sig[3], label='Acc Z')
        axes[0].set_title('Raw Accelerometer (with Gravity)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot Speed (Index 0)
        axes[1].plot(t, sig[0], label='Speed (m/s)', color='magenta')
        axes[1].set_title('GPS Speed')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot Gyroscope (Indices 4,5,6)
        axes[2].plot(t, sig[4], label='Gyro X')
        axes[2].plot(t, sig[5], label='Gyro Y')
        axes[2].plot(t, sig[6], label='Gyro Z')
        axes[2].set_title('Gyroscope (Rotational Rate)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Time (seconds)')
        
        speed_val = sig[0].mean() if sig.shape[0] > 0 else 0
        prob_pothole_str = f"| Pothole Prob: {row.get('prob_Pothole', 0.0):.3f}" if 'prob_Pothole' in row else ""
        
        fig.suptitle(f"[{idx+1}/{len(df_fn)}] FN: {row['label']} -> {row['cnn_prediction']} (Conf Non-Event: {row['cnn_confidence_pred']:.3f})\n"
                     f"Trip: {row['trip_filename']} | Speed: {speed_val:.1f} m/s {prob_pothole_str}", fontsize=12)
        
        plt.draw()

    axprev = plt.axes([0.3, 0.05, 0.15, 0.075])
    axnext = plt.axes([0.55, 0.05, 0.15, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')

    def next_fn(event):
        current_idx[0] = (current_idx[0] + 1) % len(df_fn)
        draw_plot()

    def prev_fn(event):
        current_idx[0] = (current_idx[0] - 1) % len(df_fn)
        draw_plot()

    bnext.on_clicked(next_fn)
    bprev.on_clicked(prev_fn)

    draw_plot()
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Qt5Agg') 
    main()
