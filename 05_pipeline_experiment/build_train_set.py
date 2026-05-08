import pandas as pd
import numpy as np
import os

from config import OUT_FOLDER

GT_PATH      = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
WINDOWS_PATH = os.path.join(OUT_FOLDER, "windows_features.csv")
EVENTS_PATH  = os.path.join(OUT_FOLDER, "candidates_events.csv")
OUTPUT_PATH  = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Rasio sampling background Non-Event terhadap jumlah window kelas positif
# (Pothole + Speed Bump) yang berhasil di-link.
# Contoh: BACKGROUND_RATIO=3 → jika ada 20 window Pothole+SpeedBump,
#         maka kita tambahkan hingga 60 window background Non-Event.
# Set ke None untuk menonaktifkan sampling (ambil SEMUA background yang tersedia).
BACKGROUND_RATIO = 3

# Seed untuk reproducibility
RANDOM_SEED = 42

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
# Hanya event yang sudah dilabeli secara manual
df_labeled_events = df_events.merge(
    df_gt[["event_id", "label"]],
    on="event_id",
    how="inner",
)
print(f"Events dengan label : {len(df_labeled_events)}")

if df_labeled_events.empty:
    print("\n[INFO] Belum ada event yang dilabeli. Jalankan manual_labeling_per_trip.py terlebih dahulu.")
    raise SystemExit(0)

# Trip yang sudah pernah direview oleh labeler
labeled_trip_ids = set(df_labeled_events["trip_id"].unique())
print(f"Trip yang sudah dilabeli: {len(labeled_trip_ids)}")

# ---------- STRATEGI A: TEMPORAL LINKAGE (window → event label) ----------

records_labeled = []

for trip_id, w_group in df_windows.groupby("trip_id"):
    # Hanya proses trip yang sudah direview
    if trip_id not in labeled_trip_ids:
        continue

    e_group = df_labeled_events[df_labeled_events["trip_id"] == trip_id]
    if e_group.empty:
        continue

    for _, win in w_group.iterrows():
        t0 = win["window_start"]
        t1 = win["window_end"]

        # Event berlabel yang time_s-nya jatuh di UJUNG akhir window (causal).
        # Sistem live mendeteksi event sesaat setelah terjadi, sehingga
        # event peak seharusnya berada di dekat t1 (akhir window).
        in_window = e_group[
            (e_group["time_s"] >= t1 - 0.3) & (e_group["time_s"] <= t1 + 0.1)
        ]

        if in_window.empty:
            continue  # window ini tidak mengandung event berlabel

        # Jika ada lebih dari satu event dalam window, ambil yang paling keras
        dominant = in_window.loc[in_window["peak_mag"].idxmax()]

        row = win.to_dict()
        row["label"]    = dominant["label"]
        row["event_id"] = int(dominant["event_id"])
        row["source"]   = "event_labeled"
        records_labeled.append(row)

df_event_labeled = pd.DataFrame(records_labeled) if records_labeled else pd.DataFrame()
print(f"\n[A] Window dengan label event : {len(df_event_labeled)}")
if not df_event_labeled.empty:
    print(df_event_labeled["label"].value_counts().to_string())

# ---------- STRATEGI B: BACKGROUND NON-EVENT SAMPLING ----------
# Cari window dari trip berlabel yang tidak mengandung event APAPUN
# (baik yang berlabel maupun yang belum dilabeli).
# Window semacam ini = segmen jalan normal yang aman dijadikan Non-Event.

# Bangun lookup semua event time_s per trip (berlabel DAN tidak berlabel)
# agar kita bisa mengecualikan window yang berpotensi mengandung anomali
all_events_by_trip = (
    df_events
    .groupby("trip_id")["time_s"]
    .apply(np.array)
    .to_dict()
)

records_background = []

for trip_id, w_group in df_windows.groupby("trip_id"):
    # Hanya dari trip yang sudah direview labeler
    if trip_id not in labeled_trip_ids:
        continue

    event_times = all_events_by_trip.get(trip_id, np.array([]))

    for _, win in w_group.iterrows():
        t0 = win["window_start"]
        t1 = win["window_end"]

        # Jika ada event apapun di dalam window ini, skip
        # (bahkan yang belum dilabeli — kita tidak mau mengotori kelas Non-Event)
        if len(event_times) > 0:
            has_any_event = bool(np.any((event_times >= t0) & (event_times < t1)))
        else:
            has_any_event = False

        if has_any_event:
            continue

        row = win.to_dict()
        row["label"]    = "Non-Event"
        row["event_id"] = -1          # sentinel: tidak ada event terkait
        row["source"]   = "background"
        records_background.append(row)

df_background = pd.DataFrame(records_background) if records_background else pd.DataFrame()
print(f"\n[B] Window background tersedia : {len(df_background)}")

# --- Proportional sampling agar tidak terlalu mendominasi ---
if not df_background.empty and BACKGROUND_RATIO is not None:
    # Hitung jumlah window positif (Pothole + Speed Bump)
    if not df_event_labeled.empty:
        n_positive = int(
            df_event_labeled["label"]
            .isin(["Pothole", "Speed Bump"])
            .sum()
        )
    else:
        n_positive = 0

    # Target minimum 30 background agar selalu ada representasi Non-Event
    n_target = max(n_positive * BACKGROUND_RATIO, 30)
    n_sample  = min(n_target, len(df_background))

    df_background = df_background.sample(n=int(n_sample), random_state=RANDOM_SEED)
    print(
        f"    -> Di-sample {len(df_background)} window "
        f"(target={n_target}, kelas positif={n_positive}, rasio={BACKGROUND_RATIO}x)"
    )

# ---------- GABUNGKAN A + B ----------
parts = [d for d in [df_event_labeled, df_background] if not d.empty]
if not parts:
    print("\n[INFO] Tidak ada window yang bisa digabungkan.")
    print("Pastikan timestamp di windows_features.csv dan candidates_events.csv berasal dari run yang sama.")
    raise SystemExit(0)

df_merged = pd.concat(parts, ignore_index=True)

# ---------- RINGKASAN ----------
print(f"\n{'='*50}")
print(f"Windows total dalam training set : {len(df_merged)}")
print("\nDistribusi label akhir:")
print(df_merged["label"].value_counts().to_string())

print("\nSumber data:")
print(df_merged["source"].value_counts().to_string())

# Rasio Non-Event vs positif untuk referensi
n_pos  = int(df_merged["label"].isin(["Pothole", "Speed Bump"]).sum())
n_neg  = int((df_merged["label"] == "Non-Event").sum())
ratio  = n_neg / n_pos if n_pos > 0 else float("inf")
print(f"\nRasio Non-Event : Positif = {n_neg} : {n_pos} ({ratio:.1f}x)")

# ---------- SIMPAN ----------
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Dataset siap training disimpan di: {OUTPUT_PATH}")
print(f"     Shape: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")
