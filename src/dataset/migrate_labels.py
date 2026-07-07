import os
import glob
import json
import pandas as pd
import numpy as np

# Paths
OLD_CANDIDATES_PATH = os.path.join("05_pipeline_experiment", "out", "candidates_events.csv")
NEW_CANDIDATES_PATH = os.path.join("data", "processed", "shared", "candidates_events.csv")
BACKUP_LABELS_FOLDER = os.path.join("data", "raw", "active", "labels_backup")
ACTIVE_LABELS_FOLDER = os.path.join("data", "raw", "active", "labels")

# Time tolerance for joining events (in seconds)
# Reverted to 0.5s because 0.2s caused massive loss of true positives due to filter phase shift.
# Collisions are now safely handled by priority.
TIME_TOLERANCE = 0.5

# Class priority for resolving collisions
CLASS_PRIORITY = {
    "Pothole": 3,
    "Speed Bump": 2,
    "Non-Event": 1,
    "Unsure": 0,
    "": -1
}

def get_priority(label):
    return CLASS_PRIORITY.get(label, -1)

def main():
    if not os.path.exists(OLD_CANDIDATES_PATH):
        print(f"ERROR: Old candidates not found at {OLD_CANDIDATES_PATH}")
        return
        
    if not os.path.exists(NEW_CANDIDATES_PATH):
        print(f"ERROR: New candidates not found at {NEW_CANDIDATES_PATH}")
        return

    old_df = pd.read_csv(OLD_CANDIDATES_PATH)
    new_df = pd.read_csv(NEW_CANDIDATES_PATH)
    
    print(f"Loaded {len(old_df)} old events and {len(new_df)} new events.")
    
    # Pre-calculate mapping for old events
    old_mapping = {}
    for trip_id, group in old_df.groupby("trip_id"):
        df_trip = group.copy()
        df_trip["nomor_event"] = range(1, len(df_trip) + 1)
        old_mapping[trip_id] = df_trip
        
    # Pre-calculate mapping for new events
    new_mapping = {}
    for trip_id, group in new_df.groupby("trip_id"):
        df_trip = group.copy()
        df_trip["nomor_event"] = range(1, len(df_trip) + 1)
        new_mapping[trip_id] = df_trip

    stats = {"total_labels": 0, "mapped": 0, "lost": 0, "collisions_resolved": 0}

    for backup_json_path in glob.glob(os.path.join(BACKUP_LABELS_FOLDER, "*.json")):
        with open(backup_json_path, "r") as f:
            data = json.load(f)
            
        trip_id = data.get("trip_id")
        old_labels = data.get("labels", {})
        
        if not old_labels or trip_id not in old_mapping or trip_id not in new_mapping:
            continue
            
        old_trip_df = old_mapping[trip_id]
        new_trip_df = new_mapping[trip_id]
        
        # Maps new_nomor_event -> (label_name, time_diff)
        resolved_mappings = {}
        
        for old_idx_str, label_name in old_labels.items():
            stats["total_labels"] += 1
                
            old_idx = int(old_idx_str)
            old_event_matches = old_trip_df[old_trip_df["nomor_event"] == old_idx]
            if old_event_matches.empty:
                stats["lost"] += 1
                continue
                
            old_time = old_event_matches.iloc[0]["time_s"]
            
            # Find closest new event
            time_diffs = np.abs(new_trip_df["time_s"] - old_time)
            min_diff = time_diffs.min()
            
            if min_diff <= TIME_TOLERANCE:
                best_match_idx = time_diffs.idxmin()
                new_nomor_event = str(new_trip_df.loc[best_match_idx, "nomor_event"])
                
                # Collision Detection
                if new_nomor_event in resolved_mappings:
                    existing_label, existing_diff = resolved_mappings[new_nomor_event]
                    
                    prio_new = get_priority(label_name)
                    prio_old = get_priority(existing_label)
                    
                    # Resolve collision
                    if prio_new > prio_old:
                        resolved_mappings[new_nomor_event] = (label_name, min_diff)
                        stats["collisions_resolved"] += 1
                        print(f"  [COLLISION] Overwriting {existing_label} with {label_name} (Priority Win)")
                    elif prio_new == prio_old:
                        # Tie-breaker: choose the one with smaller time difference
                        if min_diff < existing_diff:
                            resolved_mappings[new_nomor_event] = (label_name, min_diff)
                            stats["collisions_resolved"] += 1
                            print(f"  [COLLISION] Overwriting {existing_label} with {label_name} (Time Diff Win: {min_diff:.3f}s < {existing_diff:.3f}s)")
                        else:
                            print(f"  [COLLISION] Ignored {label_name} in favor of {existing_label} (Time Diff Lose: {min_diff:.3f}s >= {existing_diff:.3f}s)")
                    else:
                        print(f"  [COLLISION] Ignored {label_name} in favor of {existing_label} (Priority Lose)")
                else:
                    resolved_mappings[new_nomor_event] = (label_name, min_diff)
            else:
                stats["lost"] += 1
                
        # Build final labels dict
        new_labels = {k: v[0] for k, v in resolved_mappings.items()}
        stats["mapped"] += len(new_labels)
        
        # Save updated labels
        data["labels"] = new_labels
        data["notes"] = f"Robustly migrated from 6Hz pipeline to 1-20Hz pipeline. {len(new_labels)} labels mapped. Tolerance: {TIME_TOLERANCE}s."
        
        filename = os.path.basename(backup_json_path)
        active_json_path = os.path.join(ACTIVE_LABELS_FOLDER, filename)
        with open(active_json_path, "w") as f:
            json.dump(data, f, indent=4)
            
    print("\n" + "="*40)
    print("ROBUST MIGRATION SUMMARY")
    print("="*40)
    print(f"Total labels to map        : {stats['total_labels']}")
    print(f"Successfully mapped keys   : {stats['mapped']}")
    print(f"Lost (due to strict tol)   : {stats['lost']}")
    print(f"Collisions resolved        : {stats['collisions_resolved']}")
    print("="*40)

if __name__ == '__main__':
    main()
