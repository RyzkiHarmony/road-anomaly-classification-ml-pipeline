import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
import glob
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

from model import InceptionTime1D
from data_utils import get_stratified_group_split

def generate_chart_b64(sig, row):
    fig, axes = plt.subplots(7, 1, figsize=(4.5, 7.5), dpi=100, sharex=True)
    plt.subplots_adjust(bottom=0.08, hspace=0.6)
    
    sig_len = sig.shape[1]
    t = np.linspace(0, sig_len / 100.0, sig_len) 
    
    # 1. Acc X
    axes[0].plot(t, sig[1], color="#eab308", linewidth=0.8)
    axes[0].set_ylabel("Acc X", fontsize=6)
    axes[0].set_title('Raw Accelerometer (m/s²)', fontsize=8, pad=2)
    
    # 2. Acc Y
    axes[1].plot(t, sig[2], color="#22c55e", linewidth=0.8)
    axes[1].set_ylabel("Acc Y", fontsize=6)
    
    # 3. Acc Z
    axes[2].plot(t, sig[3], color="#3b82f6", linewidth=0.8)
    axes[2].set_ylabel("Acc Z", fontsize=6)
    
    # 4. Speed
    axes[3].plot(t, sig[0], color="magenta", linewidth=1.0)
    axes[3].set_ylabel("Speed", fontsize=6)
    axes[3].set_title('GPS Speed (m/s)', fontsize=8, pad=2)
    
    # 5. Gyro X
    axes[4].plot(t, sig[4], color="#eab308", linewidth=0.8)
    axes[4].set_ylabel("Gyr X", fontsize=6)
    axes[4].set_title('Gyroscope (rad/s)', fontsize=8, pad=2)
    
    # 6. Gyro Y
    axes[5].plot(t, sig[5], color="#22c55e", linewidth=0.8)
    axes[5].set_ylabel("Gyr Y", fontsize=6)
    
    # 7. Gyro Z
    axes[6].plot(t, sig[6], color="#3b82f6", linewidth=0.8)
    axes[6].set_ylabel("Gyr Z", fontsize=6)
    axes[6].set_xlabel('Time (s)', fontsize=7)
    
    for ax in axes:
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout(pad=0.5)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def main():
    print("Loading CNN data for False Negative map visualization...")
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    y_raw = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"))
    groups = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"))
    event_ids = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy"))
    
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    X_cnn_test = X_cnn_all[test_mask]
    y_test_raw = y_raw[test_mask]
    groups_test = groups[test_mask]
    event_ids_test = event_ids[test_mask]
    
    df_test = pd.DataFrame({
        'label': y_test_raw,
        'trip_id': groups_test,
        'event_id': event_ids_test
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
        
    if "Speed Bump" in list(classes):
        sb_idx = list(classes).index("Speed Bump")
        df_test['prob_SpeedBump'] = cnn_probas[:, sb_idx]
        
    # Load XGBoost CSV to get lat/lon for these events
    xgb_df_path = os.path.join(XGB_DATA_DIR, "xgboost_labeled_windows.csv")
    df_xgb = pd.read_csv(xgb_df_path)
    # create mapping from event_id to (lat, lon, time_s)
    # assuming event_id is numeric and matches between the two
    xgb_meta = df_xgb.set_index('event_id')[['lat', 'lon', 'time_s']].to_dict('index')
    
    df_test['lat'] = df_test['event_id'].apply(lambda x: xgb_meta.get(x, {}).get('lat', np.nan))
    df_test['lon'] = df_test['event_id'].apply(lambda x: xgb_meta.get(x, {}).get('lon', np.nan))
    df_test['time_s'] = df_test['event_id'].apply(lambda x: xgb_meta.get(x, {}).get('time_s', np.nan))
    
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
            
    df_test['trip_filename'] = df_test['trip_id'].map(lambda x: trip_to_filename.get(x, 'Unknown Trip'))
    df_test['event_in_trip'] = df_test.groupby('trip_id')['time_s'].rank(method='first', na_option='bottom').astype(int)
    
    fn_mask = (df_test['label'].isin(['Pothole', 'Speed Bump'])) & (df_test['cnn_prediction'] == 'Non-Event')
    df_fn = df_test[fn_mask].copy()
    X_fn = X_cnn_test[fn_mask]
    
    # Drop rows without lat/lon
    valid_loc_mask = ~df_fn['lat'].isna() & ~df_fn['lon'].isna()
    df_fn = df_fn[valid_loc_mask]
    X_fn = X_fn[valid_loc_mask]
    
    sort_idx = np.argsort(-df_fn['cnn_confidence_pred'].values)
    df_fn = df_fn.iloc[sort_idx].reset_index(drop=True)
    X_fn = X_fn[sort_idx]
    
    if len(df_fn) == 0:
        print("No False Negatives with valid locations found!")
        return
        
    print(f"Found {len(df_fn)} False Negatives with locations. Building map...")
    
    center_lat = df_fn["lat"].mean()
    center_lon = df_fn["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
    
    # Plot trips routes
    fn_trips = df_fn['trip_id'].unique()
    for trip_id in fn_trips:
        trip_coords = df_xgb[(df_xgb['trip_id'] == trip_id) & (~df_xgb['lat'].isna())].sort_values('time_s')[['lat', 'lon']].values.tolist()
        if len(trip_coords) > 0:
            folium.PolyLine(trip_coords, color="blue", weight=3, opacity=0.3).add_to(m)

    for i, row in df_fn.iterrows():
        lat = row['lat']
        lon = row['lon']
        sig = X_fn[i]
        speed_val = sig[0].mean() if sig.shape[0] > 0 else 0
        b64 = generate_chart_b64(sig, row)
        chart_html = f'<img src="data:image/png;base64,{b64}" style="width:100%;margin-top:6px;">'
        
        prob_pothole = row.get('prob_Pothole', 0.0)
        prob_sb = row.get('prob_SpeedBump', 0.0)
        
        color = "red" if row['label'] == "Pothole" else "orange"
        
        popup_html = f'''
        <div style="width:340px; font-family:sans-serif;">
            <b>Event ID:</b> {row['event_id']}<br>
            <b>Event ID on Trip:</b> {row['event_in_trip']}<br>
            <b>Trip ID:</b> {row['trip_id']}<br>
            <b>Trip File:</b> {row['trip_filename']}<br>
            <b>Actual:</b> {row['label']}<br>
            <b>Prediction:</b> {row['cnn_prediction']}<br>
            <b>Confidence (Non-Event):</b> {row['cnn_confidence_pred']:.2f}<br>
            <b>Pothole Prob:</b> {prob_pothole:.2f}<br>
            <b>Speed Bump Prob:</b> {prob_sb:.2f}<br>
            <b>Speed:</b> {speed_val:.1f} m/s<br>
            {chart_html}
        </div>
        '''
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=380),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)

    map_path = os.path.join(REPORT_DIR, "false_negatives_map.html")
    m.save(map_path)
    print(f"Map saved successfully to {map_path}")

if __name__ == "__main__":
    main()
