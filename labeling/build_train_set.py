# build_train_set.py
#
# Menggabungkan label manual (ground_truth_labels.csv) dengan fitur windows
# (windows_features.csv) menggunakan linkage temporal berbasis event.
#
# Strategi linkage:
#   Sebuah window mendapat label dari event jika:
#     1. Keduanya berasal dari trip_id yang sama, DAN
#     2. event.time_s berada di dalam rentang [window_start, window_end]
#
#   Satu window bisa mencakup lebih dari satu event; dalam kasus itu diambil
#   label dari event dengan peak_mag tertinggi di dalam window tersebut
#   (dominant-event assignment).

import pandas as pd
import os

_DIR       = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUT_FOLDER = os.path.join(_DIR, "out")

GT_PATH       = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
WINDOWS_PATH  = os.path.join(OUT_FOLDER, "windows_features.csv")
EVENTS_PATH   = os.path.join(OUT_FOLDER, "candidates_events.csv")
OUTPUT_PATH   = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")

# ---------- LOAD ----------
missing = [p for p in (GT_PATH, WINDOWS_PATH, EVENTS_PATH) if not os.path.exists(p)]
if missing:
    print("File berikut belum ditemukan:")
    for p in missing:
        print(f"  {p}")
    raise SystemExit(1)

df_gt      = pd.read_csv(GT_PATH)       # event_id, label, trip_id, labeled_at
df_windows = pd.read_csv(WINDOWS_PATH)  # window_start, window_end, trip_id, features...
df_events  = pd.read_csv(EVENTS_PATH)   # event_id, time_s, peak_mag, trip_id, ...

print(f"Ground truth rows   : {len(df_gt)}")
print(f"Windows             : {len(df_windows)}")
print(f"Candidate events    : {len(df_events)}")

# ---------- ATTACH LABEL TO EVENTS ----------
# Hanya simpan event yang sudah dilabeli secara manual
df_labeled_events = df_events.merge(
    df_gt[["event_id", "label"]],
    on="event_id",
    how="inner",
)
print(f"Events dengan label : {len(df_labeled_events)}")

if df_labeled_events.empty:
    print("\n[INFO] Belum ada event yang dilabeli. Jalankan manual_labeling_per_trip.py terlebih dahulu.")
    raise SystemExit(0)

# ---------- TEMPORAL LINKAGE ----------
# Untuk setiap window, cari event berlabel yang time_s-nya jatuh di dalam
# [window_start, window_end] pada trip yang sama.

records = []

for trip_id, w_group in df_windows.groupby("trip_id"):
    # Event berlabel untuk trip ini saja
    e_group = df_labeled_events[df_labeled_events["trip_id"] == trip_id]

    if e_group.empty:
        continue  # trip ini belum ada labelnya, lewati

    for _, win in w_group.iterrows():
        t0 = win["window_start"]
        t1 = win["window_end"]

        # Event yang time_s-nya jatuh dalam rentang window ini
        in_window = e_group[
            (e_group["time_s"] >= t0) & (e_group["time_s"] < t1)
        ]

        if in_window.empty:
            continue  # window ini tidak mengandung event berlabel

        # Jika ada lebih dari satu event dalam window, ambil yang paling keras
        dominant = in_window.loc[in_window["peak_mag"].idxmax()]

        row = win.to_dict()
        row["label"]    = dominant["label"]
        row["event_id"] = int(dominant["event_id"])
        records.append(row)

if not records:
    print("\n[INFO] Tidak ada window yang beririsan dengan event berlabel.")
    print("Pastikan timestamp di windows_features.csv dan candidates_events.csv berasal dari run yang sama.")
    raise SystemExit(0)

df_merged = pd.DataFrame(records)

# ---------- RINGKASAN ----------
print(f"\nWindows yang berhasil dilabeli : {len(df_merged)}")
print("Distribusi label:")
print(df_merged["label"].value_counts().to_string())

# ---------- SIMPAN ----------
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Dataset siap training disimpan di: {OUTPUT_PATH}")
print(f"     Shape: {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")
