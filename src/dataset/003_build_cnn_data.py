import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(__file__))

from config import CNN_OUT_DIR, CSV_FOLDER, OUT_FOLDER, TARGET_HZ, WINDOW_SIZE_S, get_logger
from cnn_dataset_utils import (
    BACKGROUND_RATIO,
    CHANNELS,
    CNN_DATA_DIR,
    EVENT_GAP_BUFFER_S,
    EVENT_WINDOW_HALF_S,
    EVENTS_PATH,
    EXTENDED_SEQ_LEN,
    GAP_GUARD_BAND_S,
    GT_PATH,
    MAX_JITTER_SAMPLES,
    SEQ_LEN,
    _event_overlaps_buffered_gap,
    _require_columns,
    compute_engineered_features,
    extract_sequence,
    find_large_timestamp_gaps,
    get_csv_path_for_trip,
    resample_100hz,
    split_contiguous_segments,
)

logger = get_logger(__name__)

def main():
    if not os.path.exists(GT_PATH) or not os.path.exists(EVENTS_PATH):
        logger.error("File ground_truth_labels.csv atau candidates_events.csv tidak ditemukan.")
        return

    df_gt = pd.read_csv(GT_PATH)
    df_events = pd.read_csv(EVENTS_PATH)

    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")

    shared_bg_path = os.path.join(OUT_FOLDER, "shared_background.csv")
    # [DOKUMENTASI]: shared_background.csv adalah sampel kelas Non-Event yang di-generate
    # secara acak dari trip yang sama dengan data ground truth untuk menyeimbangkan kelas.
    # Karena trip_id dipertahankan aslinya, pembagian StratifiedGroupKFold berdasarkan
    # trip_id di train.py memastikan sampel background ini TIDAK menyebabkan data leakage
    # lintas set (train vs test).
    if os.path.exists(shared_bg_path):
        try:
            df_bg = pd.read_csv(shared_bg_path)
            if not df_bg.empty:
                df_labeled = pd.concat([df_labeled, df_bg], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass

    df_labeled = df_labeled.sort_values(["trip_id", "time_s"]).reset_index(drop=True)

    if df_labeled.empty:
        logger.warning("Belum ada data yang dilabeli.")
        return

    X_list = []
    y_list = []
    groups_list = []
    event_ids_list = []
    skipped_events = 0
    skipped_by_trip = {}
    skipped_by_reason = {
        "gap_buffer": 0,
        "extract_failed": 0,
        "preprocess_failed": 0,
        "missing_csv": 0,
    }

    grouped_by_trip = df_labeled.groupby("trip_id")
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            logger.info(f"Processing trip {trip_id}...")
            try:
                raw_df = pd.read_csv(csv_path)
                if 'speed' not in raw_df.columns:
                    raw_df['speed'] = 0.0
                raw_df = compute_engineered_features(raw_df)
                gap_intervals = find_large_timestamp_gaps(raw_df)
                segments = split_contiguous_segments(raw_df)

                if gap_intervals:
                    logger.info(
                        f"Trip {trip_id} has {len(gap_intervals)} large gap(s); applying {EVENT_GAP_BUFFER_S:.2f}s event buffer"
                    )

                for segment_idx, segment_raw in enumerate(segments):
                    segment_start_s = float(segment_raw["timestamp"].iloc[0]) / 1000.0
                    segment_end_s = float(segment_raw["timestamp"].iloc[-1]) / 1000.0
                    segment_events = group[(group["time_s"] >= segment_start_s) & (group["time_s"] <= segment_end_s)]

                    try:
                        segment_df = resample_100hz(segment_raw)
                    except Exception as e:
                        logger.warning(f"Trip {trip_id} segment {segment_idx} failed to resample: {e}")
                        skipped_events += len(segment_events)
                        skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(segment_events)
                        skipped_by_reason["preprocess_failed"] += len(segment_events)
                        continue

                    if len(segment_df) < EXTENDED_SEQ_LEN:
                        continue

                    for _, row in segment_events.iterrows():
                        t_event = float(row["time_s"])

                        overlaps_gap, _distance_to_gap = _event_overlaps_buffered_gap(t_event, gap_intervals)
                        if overlaps_gap:
                            skipped_events += 1
                            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + 1
                            skipped_by_reason["gap_buffer"] += 1
                            continue

                        seq = extract_sequence(segment_df, t_event)

                        if seq is not None:
                            X_list.append(seq)
                            y_list.append(row["label"])
                            groups_list.append(trip_id)
                            event_ids_list.append(row["event_id"])
                        else:
                            skipped_events += 1
                            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + 1
                            skipped_by_reason["extract_failed"] += 1
            except Exception as e:
                logger.error(f"Failed to process trip {trip_id}: {e}")
                skipped_events += len(group)
                skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(group)
        else:
            skipped_events += len(group)
            skipped_by_trip[trip_id] = skipped_by_trip.get(trip_id, 0) + len(group)
            skipped_by_reason["missing_csv"] += len(group)


    if not X_list:
        raise ValueError("No CNN samples were extracted. Check raw CSV schema and preprocessing rules.")


    X = np.stack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    event_ids = np.array(event_ids_list)

    # Transpose X to (N_samples, Channels, Length) for PyTorch 1D-CNN
    X = np.transpose(X, (0, 2, 1))

    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"), X)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"), y)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"), groups)
    np.save(os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy"), event_ids)

    logger.info(f"Dataset 1D-CNN disimpan. Shape X: {X.shape}, Shape y: {y.shape}")
    if skipped_events:
        top_skipped = sorted(skipped_by_trip.items(), key=lambda item: item[1], reverse=True)[:5]
        logger.warning(f"Skipped {skipped_events} CNN samples during extraction.")
        logger.warning("Top skipped trips: " + ", ".join([f"{trip_id}:{count}" for trip_id, count in top_skipped]))
        logger.warning(
            "Skip reasons: "
            + ", ".join([f"{reason}={count}" for reason, count in skipped_by_reason.items()])
        )

if __name__ == "__main__":
    main()
