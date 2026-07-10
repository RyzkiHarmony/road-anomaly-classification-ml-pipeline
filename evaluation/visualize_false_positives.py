import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
import datetime
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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
    print("Loading data for visualization...")
    xgb_df_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(xgb_df_path).dropna(subset=['label'])
    
    y_raw = df['label'].values
    groups = df['trip_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))
        
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_mask]])
    
    df_test = df[test_mask][is_original_test].copy()
    
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    X_cnn_test = X_cnn_all[test_mask][is_original_test]
    
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    classes = le.classes_
    scaler_path = os.path.join(CNN_MODEL_DIR, "cnn_1d_scaler_params.json")
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    global_means = np.array(scaler_params['means']).reshape(1, 7, 1)
    global_stds = np.array(scaler_params['stds']).reshape(1, 7, 1)
    X_cnn_scaled = (X_cnn_test - global_means) / global_stds
    
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    cnn_model = InceptionTime1D(in_channels=X_cnn_tensor.shape[1], num_classes=len(classes))
    cnn_model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    cnn_model.eval()
    
    with torch.no_grad():
        outputs = cnn_model(X_cnn_tensor)
        cnn_probas = torch.softmax(outputs, dim=1).numpy()
        
    cnn_preds = np.argmax(cnn_probas, axis=1)
    df_test['cnn_prediction'] = le.inverse_transform(cnn_preds)
    df_test['cnn_confidence'] = np.max(cnn_probas, axis=1)
    
    
    # Map trip_id to filename and start time using metadata
    meta_dir = os.path.join(BASE_DIR, "data", "raw", "active", "meta")
    trip_to_filename = {}
    trip_to_start_time = {}
    for meta_file in glob.glob(os.path.join(meta_dir, "*.json")):
        try:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
                if 'tripId' in meta_data:
                    base_name = os.path.basename(meta_file).replace('.json', '')
                    trip_to_filename[meta_data['tripId']] = base_name
                    if 'startTime' in meta_data:
                        trip_to_start_time[meta_data['tripId']] = meta_data['startTime'] / 1000.0
        except Exception as e:
            pass
            
    unique_trips = df_test['trip_id'].unique()
    df_test['trip_index'] = df_test['trip_id'].apply(lambda x: np.where(unique_trips == x)[0][0] + 1)
    df_test['event_in_trip'] = df_test.groupby('trip_id')['time_s'].rank(method='first').astype(int)
    
    def format_time(row):
        # Fallback to the row's own time if trip start time is unknown
        start_time = trip_to_start_time.get(row['trip_id'], row['time_s'])
        rel_time = row['time_s'] - start_time
        if rel_time < 0: rel_time = 0
        m = int(rel_time // 60)
        s = int(rel_time % 60)
        return f"menit {m} detik {s}"
        
    df_test['time_formatted'] = df_test.apply(format_time, axis=1)
    df_test['trip_filename'] = df_test['trip_id'].map(lambda x: trip_to_filename.get(x, 'Unknown Trip'))
    
    # Filter FP
    fp_mask = (df_test['label'] == 'Non-Event') & (df_test['cnn_prediction'].isin(['Pothole', 'Speed Bump']))
    
    df_fp = df_test[fp_mask].copy()
    X_fp = X_cnn_test[fp_mask] # Raw unscaled data for visualization
    
    # Sort by confidence
    sort_idx = np.argsort(-df_fp['cnn_confidence'].values)
    df_fp = df_fp.iloc[sort_idx].reset_index(drop=True)
    X_fp = X_fp[sort_idx]
    
    if len(df_fp) == 0:
        print("No False Positives found!")
        return

    print(f"Found {len(df_fp)} False Positives. Launching interactive visualizer...")
    
    # Visualization UI Setup
    current_idx = [0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    # Channels definition from build_cnn_data.py
    # 0: speed, 1-3: ax,ay,az, 4-6: gx,gy,gz, 7-9: lin_ax,lin_ay,lin_az, 10-12: grav_x,grav_y,grav_z
    time_axis = np.linspace(0, 2.0, 200) # assuming 200 samples = 2 seconds
    
    def draw_plot():
        for ax in axes:
            ax.clear()
            
        idx = current_idx[0]
        row = df_fp.iloc[idx]
        sig = X_fp[idx] # Shape: (13, 230) -> crop to 200 if needed, usually we visualize the first 200 or the whole 230
        
        sig_len = sig.shape[1]
        t = np.linspace(0, sig_len / 100.0, sig_len) # Assuming 100Hz
        
        # Plot Linear Accel (Indices 7,8,9)
        axes[0].plot(t, sig[7], label='Lin Accel X')
        axes[0].plot(t, sig[8], label='Lin Accel Y')
        axes[0].plot(t, sig[9], label='Lin Accel Z')
        axes[0].set_title('Linear Acceleration (Without Gravity)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot Gravity (Indices 10,11,12)
        axes[1].plot(t, sig[10], label='Grav X')
        axes[1].plot(t, sig[11], label='Grav Y')
        axes[1].plot(t, sig[12], label='Grav Z')
        axes[1].set_title('Gravity Vector (Orientation)')
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
        
        # Super title
        speed_val = sig[0].mean() if sig.shape[0] > 0 else 0
        fig.suptitle(f"[{idx+1}/{len(df_fp)}] FP: Non-Event -> {row['cnn_prediction']} (Conf: {row['cnn_confidence']:.3f})\n"
                     f"Trip: {row['trip_filename']} | Event #{row['event_in_trip']} in Trip | Time: {row['time_formatted']} | Speed: {speed_val:.1f} m/s", fontsize=12)
        
        plt.draw()

    # Buttons
    axprev = plt.axes([0.3, 0.05, 0.15, 0.075])
    axnext = plt.axes([0.55, 0.05, 0.15, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')

    def next_fp(event):
        current_idx[0] = (current_idx[0] + 1) % len(df_fp)
        draw_plot()

    def prev_fp(event):
        current_idx[0] = (current_idx[0] - 1) % len(df_fp)
        draw_plot()

    bnext.on_clicked(next_fp)
    bprev.on_clicked(prev_fp)

    draw_plot()
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Qt5Agg') # Ensure an interactive backend is used on Windows
    main()
