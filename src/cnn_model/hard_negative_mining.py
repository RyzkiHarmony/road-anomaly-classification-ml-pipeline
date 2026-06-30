import os
import glob
import torch
import numpy as np
import pandas as pd
from model import InceptionTime1D

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_pipeline_experiment'))
from config import CSV_FOLDER, OUT_FOLDER, WINDOW_SIZE_S, TARGET_HZ, get_logger
from sensor_fusion import apply_sensor_fusion

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "cnn_1d")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d")

SEQ_LEN = int(WINDOW_SIZE_S * TARGET_HZ)  # 200
STRIDE = 50  # 0.5s stride
CHANNELS = [
    "a_vertical", "a_horizontal", "speed", 
    "a_vertical_crest_factor", "a_vertical_jerk",
    "gx", "gy", "gz", 
    "g_roll_accel", "g_pitch_accel",
    "a_vertical_rms", "a_vertical_zcr",
    "a_horizontal_rms", "energy_ratio_vh",
    "lin_ax", "lin_ay", "lin_az"
]

CONF_THRESHOLD = 0.50

def compute_engineered_features(df):
    """
    Menghitung fitur tambahan:
    1. Crest Factor dari akselerasi vertikal (a_vertical_crest_factor)
    2. Jerk dari akselerasi vertikal (a_vertical_jerk)
    3. Turunan gyro: Roll acceleration (da_roll/dt) & Pitch acceleration (da_pitch/dt)
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    dt = df["timestamp"].diff().fillna(10.0) / 1000.0  # interval default 10ms
    dt = np.where(dt <= 0, 0.01, dt)
    
    # 1. Jerk
    jerk = df["a_vertical"].diff().fillna(0.0) / dt
    df["a_vertical_jerk"] = jerk
    
    # 2. Crest Factor
    window_sz = 10
    peak = df["a_vertical"].abs().rolling(window=window_sz, min_periods=1, center=True).max()
    rms = np.sqrt((df["a_vertical"]**2).rolling(window=window_sz, min_periods=1, center=True).mean())
    crest_factor = peak / (rms + 1e-6)
    df["a_vertical_crest_factor"] = crest_factor.fillna(1.0)
    
    # 3. Gyro derivatives (ang. acceleration)
    # Pastikan kolom gx, gy, gz ada di dataframe
    for col in ["gx", "gy", "gz"]:
        if col not in df.columns:
            df[col] = 0.0
            
    df["g_roll_accel"] = df["gx"].diff().fillna(0.0) / dt
    df["g_pitch_accel"] = df["gy"].diff().fillna(0.0) / dt
    
    # 4. Speed-Normalized Acceleration & Jerk
    speed_safe = df["speed"].clip(lower=0).fillna(0.0) + 0.5
    df["a_vertical_speed_norm"] = df["a_vertical"] / speed_safe
    df["jerk_speed_norm"] = df["a_vertical_jerk"] / speed_safe
    
    # 5. Rolling RMS (Root Mean Square) dari akselerasi vertikal
    rms_window = 20
    df["a_vertical_rms"] = np.sqrt(
        (df["a_vertical"]**2).rolling(window=rms_window, min_periods=1, center=True).mean()
    ).fillna(0.0)
    
    # 6. Zero Crossing Rate (ZCR) dari akselerasi vertikal
    zcr_window = 20
    sign_changes = (np.sign(df["a_vertical"]).diff().abs() > 0).astype(float)
    df["a_vertical_zcr"] = sign_changes.rolling(
        window=zcr_window, min_periods=1, center=True
    ).mean().fillna(0.0)
    
    # 7. Rolling RMS dari akselerasi horizontal
    df["a_horizontal_rms"] = np.sqrt(
        (df["a_horizontal"]**2).rolling(window=rms_window, min_periods=1, center=True).mean()
    ).fillna(0.0)
    
    # 8. Rasio Energi Vertikal / Horizontal
    df["energy_ratio_vh"] = df["a_vertical_rms"] / (df["a_horizontal_rms"] + 1e-6)
    
    return df

def get_trip_id_from_csv(csv_path):
    # Nama file: RoadDamage_2026-05-26_15-19-36.csv
    # Kita butuh baca metadata untuk trip_id
    base = os.path.basename(csv_path).replace(".csv", ".json")
    meta_path = os.path.join(os.path.dirname(CSV_FOLDER), "meta", base)
    if os.path.exists(meta_path):
        import json
        with open(meta_path, 'r') as f:
            try:
                meta = json.load(f)
                return meta.get("tripId", None)
            except:
                pass
    return None

def main():
    pth_path = os.path.join(MODEL_DIR, "cnn_1d_model.pth")
    classes_path = os.path.join(MODEL_DIR, "cnn_1d_classes.npy")
    
    if not os.path.exists(pth_path):
        logger.error("Model tidak ditemukan. Harus ditraining dulu.")
        return
        
    classes = np.load(classes_path)
    p_idx = list(classes).index("Pothole")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime1D(in_channels=18, num_classes=len(classes),
                           num_blocks=2, channels=64, bottleneck_channels=16, dropout_rate=0.2).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Load Events GT
    EVENTS_PATH = os.path.join(OUT_FOLDER, "candidates_events.csv")
    df_events = pd.read_csv(EVENTS_PATH) if os.path.exists(EVENTS_PATH) else pd.DataFrame()
    
    hard_negatives_X = []
    hard_negatives_groups = []
    
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    logger.info(f"Scanning {len(csv_files)} trips untuk mencari Hard Negatives...")
    
    for csv_path in csv_files:
        trip_id = get_trip_id_from_csv(csv_path)
        if not trip_id: continue
        
        event_times = df_events[df_events["trip_id"] == trip_id]["time_s"].values if not df_events.empty else np.array([])
        
        try:
            raw_df = pd.read_csv(csv_path)
            raw_df = apply_sensor_fusion(raw_df)
            if "speed" not in raw_df.columns:
                raw_df["speed"] = 0.0
            raw_df = compute_engineered_features(raw_df)
                
            arr = raw_df[CHANNELS].fillna(0.0).values
            times = raw_df["timestamp"].values / 1000.0
            
            n_samples = len(arr)
            if n_samples < SEQ_LEN: continue
            
            # Buat batch windows
            windows = []
            window_times = []
            
            for start_idx in range(0, n_samples - SEQ_LEN + 1, STRIDE):
                windows.append(arr[start_idx:start_idx+SEQ_LEN])
                window_times.append(times[start_idx + SEQ_LEN//2])
                
            if len(windows) == 0: continue
            
            # (N, 200, 3) -> (N, 3, 200)
            windows_np = np.stack(windows)
            windows_np = np.transpose(windows_np, (0, 2, 1))
            
            # Inferensi per batch 1024 agar tidak OOM
            windows_tensor = torch.tensor(windows_np, dtype=torch.float32)
            
            from torch.utils.data import TensorDataset, DataLoader
            loader = DataLoader(TensorDataset(windows_tensor), batch_size=1024, shuffle=False)
            
            all_probs = []
            with torch.no_grad():
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    
            all_probs = np.concatenate(all_probs, axis=0)
            
            # Cari yang Pothole conf > CONF_THRESHOLD
            pothole_probs = all_probs[:, p_idx]
            fp_indices = np.where(pothole_probs > CONF_THRESHOLD)[0]
            
            for idx in fp_indices:
                t_center = window_times[idx]
                
                # Cek jarak ke event manual (baik pothole, bump, dsb)
                if len(event_times) > 0:
                    dist = np.min(np.abs(event_times - t_center))
                    if dist < 3.0: 
                        # Terlalu dekat dengan ground truth, skip
                        continue
                        
                # Filter speed mati
                if windows_np[idx][2].mean() < 2.0:
                    continue
                    
                # Sah! Ini Hard Negative murni.
                hard_negatives_X.append(windows_np[idx])
                hard_negatives_groups.append(trip_id)
                
        except Exception as e:
            logger.error(f"Failed on {csv_path}: {e}")
            
    if len(hard_negatives_X) > 0:
        hn_X = np.stack(hard_negatives_X)
        hn_y = np.array(["Non-Event"] * len(hard_negatives_X))
        hn_g = np.array(hard_negatives_groups)
        
        logger.info(f"Ditemukan {len(hn_X)} Hard Negatives!")
        
        # Karantina data hard negatives ke file terpisah untuk divalidasi secara manual
        hn_X_path = os.path.join(DATA_DIR, "X_hard_negatives.npy")
        hn_y_path = os.path.join(DATA_DIR, "y_hard_negatives.npy")
        hn_g_path = os.path.join(DATA_DIR, "groups_hard_negatives.npy")
        
        np.save(hn_X_path, hn_X)
        np.save(hn_y_path, hn_y)
        np.save(hn_g_path, hn_g)
        
        logger.info(f"Hard negatives berhasil disimpan di karantina: {hn_X_path}")
        logger.warning("PENTING: Tinjau sampel hard negatives di atas secara manual sebelum digabungkan ke dataset utama untuk mencegah Label Contamination!")
    else:
        logger.info("Tidak ada Hard Negative baru yang ditemukan. Model sudah sempurna atau threshold terlalu tinggi.")

if __name__ == "__main__":
    main()
