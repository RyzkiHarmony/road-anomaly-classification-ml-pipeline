import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tambahkan path ke utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import OUT_FOLDER, get_logger, TARGET_HZ

from build_cnn_data import (
    GT_PATH, EVENTS_PATH, EXTENDED_SEQ_LEN, EVENT_WINDOW_HALF_S, 
    GAP_GUARD_BAND_S, EVENT_GAP_BUFFER_S, CHANNELS,
    compute_engineered_features, get_csv_path_for_trip, 
    find_large_timestamp_gaps, split_contiguous_segments, 
    _event_overlaps_buffered_gap, extract_sequence, resample_100hz
)

logger = get_logger("visualize_defect_windows")
DEFECT_OUT_DIR = os.path.join(OUT_FOLDER, "defect_windows")
os.makedirs(DEFECT_OUT_DIR, exist_ok=True)

def plot_defect(raw_df, t_center, event_id, label, reason, gap_intervals=None):
    """Memvisualisasikan data mentah di sekitar titik kejadian yang defek."""
    # Ambil jendela waktu mentah +- 3 detik untuk visualisasi (lebih lebar dari window 2.3 detik asli)
    view_half_s = 3.0
    times = raw_df["timestamp"].astype(float).values / 1000.0
    
    idx_start = np.searchsorted(times, t_center - view_half_s)
    idx_end = np.searchsorted(times, t_center + view_half_s)
    
    if idx_start >= len(raw_df) or idx_end <= 0 or idx_start == idx_end:
        logger.warning(f"Event {event_id}: Tidak ada data raw di sekitar waktu {t_center:.2f}s")
        return

    seg = raw_df.iloc[idx_start:idx_end]
    seg_times = seg["timestamp"].astype(float).values / 1000.0
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Defect Window Analysis - Event ID: {event_id} | Label: {label}\nReason: {reason}", fontsize=14)

    # 1. Plot Accelerometer (Tampilkan titik asli untuk melihat kekosongan)
    if all(c in seg.columns for c in ['ax', 'ay', 'az']):
        axs[0].plot(seg_times, seg['ax'], 'r.-', label='ax', markersize=4, linewidth=1)
        axs[0].plot(seg_times, seg['ay'], 'g.-', label='ay', markersize=4, linewidth=1)
        axs[0].plot(seg_times, seg['az'], 'b.-', label='az', markersize=4, linewidth=1)
    axs[0].set_ylabel("Accel (m/s^2)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # Gambarkan batas window asli (2.3 detik)
    axs[0].axvspan(t_center - EVENT_WINDOW_HALF_S, t_center + EVENT_WINDOW_HALF_S, color='yellow', alpha=0.2, label="Target Extract Window")
    axs[0].axvline(t_center, color='k', linestyle='-', lw=2, label='Event Center')

    # 2. Plot Timestamp Deltas (Untuk mendeteksi Gap secara visual)
    deltas_ms = np.diff(seg["timestamp"].astype(float).values)
    # Tambahkan 0 di awal agar panjang sama dengan seg_times
    deltas_ms = np.insert(deltas_ms, 0, 0) 
    
    axs[1].bar(seg_times, deltas_ms, color='orange', width=0.05, alpha=0.7, label='Time Delta (ms)')
    axs[1].axhline(10, color='g', linestyle='--', label='Ideal Delta (10ms)')
    axs[1].axhline(50, color='r', linestyle='--', label='Max Interpolation Limit (50ms)')
    axs[1].set_ylabel("Delta Antar Sampel (ms)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Gambarkan gap_intervals jika ada
    if gap_intervals:
        for gap in gap_intervals:
            gs = gap["start_ms"] / 1000.0
            ge = gap["end_ms"] / 1000.0
            if ge >= (t_center - view_half_s) and gs <= (t_center + view_half_s):
                axs[1].axvspan(gs, ge, color='red', alpha=0.3, label="Detected Large Gap")

    # 3. Plot Gyroscope
    if all(c in seg.columns for c in ['gx', 'gy', 'gz']):
        axs[2].plot(seg_times, seg['gx'], 'c.-', label='gx', markersize=4, linewidth=1)
        axs[2].plot(seg_times, seg['gy'], 'm.-', label='gy', markersize=4, linewidth=1)
        axs[2].plot(seg_times, seg['gz'], 'y.-', label='gz', markersize=4, linewidth=1)
    axs[2].set_ylabel("Gyro (rad/s)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    safe_reason = reason.replace(" ", "_")
    filename = f"defect_{event_id}_{safe_reason}.png"
    plt.savefig(os.path.join(DEFECT_OUT_DIR, filename), dpi=150)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File GT atau Events tidak ditemukan.")
        sys.exit(1)

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    df_labeled = df_labeled.sort_values(["trip_id", "time_s"]).reset_index(drop=True)
    
    logger.info(f"Memulai pelacakan defect pada {len(df_labeled)} kandidat kejadian...")

    grouped_by_trip = df_labeled.groupby("trip_id")
    defect_count = 0
    
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if not csv_path:
            continue
            
        try:
            raw_df = pd.read_csv(csv_path)
            if 'speed' not in raw_df.columns:
                raw_df['speed'] = 0.0
            raw_df = compute_engineered_features(raw_df)
            gap_intervals = find_large_timestamp_gaps(raw_df)
            segments = split_contiguous_segments(raw_df)

            for segment_idx, segment_raw in enumerate(segments):
                segment_start_s = float(segment_raw["timestamp"].iloc[0]) / 1000.0
                segment_end_s = float(segment_raw["timestamp"].iloc[-1]) / 1000.0
                segment_events = group[(group["time_s"] >= segment_start_s) & (group["time_s"] <= segment_end_s)]

                try:
                    segment_df = resample_100hz(segment_raw)
                except Exception as e:
                    # Seluruh segment gagal di-resample (biasanya durasi terlalu pendek)
                    for _, row in segment_events.iterrows():
                        t_event = float(row["time_s"])
                        plot_defect(raw_df, t_event, row["event_id"], row["label"], "Segment Too Short", gap_intervals)
                        defect_count += 1
                    continue

                if len(segment_df) < EXTENDED_SEQ_LEN:
                    # Segment terlalu pendek untuk mengekstrak 1 window penuh (230 samples)
                    for _, row in segment_events.iterrows():
                        t_event = float(row["time_s"])
                        plot_defect(raw_df, t_event, row["event_id"], row["label"], "Segment Too Short", gap_intervals)
                        defect_count += 1
                    continue

                for _, row in segment_events.iterrows():
                    t_event = float(row["time_s"])

                    # 1. Cek Gap Buffer
                    overlaps_gap, _ = _event_overlaps_buffered_gap(t_event, gap_intervals)
                    if overlaps_gap:
                        plot_defect(raw_df, t_event, row["event_id"], row["label"], "Overlaps Gap Buffer", gap_intervals)
                        defect_count += 1
                        continue

                    # 2. Cek Ekstraksi
                    seq = extract_sequence(segment_df, t_event)
                    if seq is None:
                        # Extract_sequence mereturn None jika coverage < 70% atau ada NaN
                        plot_defect(raw_df, t_event, row["event_id"], row["label"], "Extract Failed (NaN or Coverage)", gap_intervals)
                        defect_count += 1

        except Exception as e:
            logger.error(f"Error memproses trip {trip_id}: {e}")

    logger.info(f"Selesai! Ditemukan dan divisualisasikan {defect_count} defect windows.")
    logger.info(f"Hasil visualisasi disimpan di: {DEFECT_OUT_DIR}")
