# %%
import pandas as pd
import numpy as np
import folium
import os
import glob
import json
import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, aman untuk batch rendering
import matplotlib.pyplot as plt

_DIR        = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUT_FOLDER  = os.path.join(_DIR, "out")
CSV_FOLDER  = os.path.join(_DIR, "data", "csv")
META_FOLDER = os.path.join(_DIR, "data", "meta")

CANDIDATE_PATH = os.path.join(OUT_FOLDER, "candidates_events.csv")
MASTER_GT_PATH = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")

# ---------- HELPER: map trip_id -> raw CSV path ----------

def build_trip_csv_map():
    """Scan meta JSON files to build a {trip_id: csv_path} lookup."""
    mapping = {}
    for json_path in glob.glob(os.path.join(META_FOLDER, "*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        trip_id = meta.get("tripId")
        if not trip_id:
            continue
        csv_name = os.path.basename(json_path).rsplit(".", 1)[0] + ".csv"
        csv_path = os.path.join(CSV_FOLDER, csv_name)
        if os.path.exists(csv_path):
            mapping[trip_id] = csv_path
    return mapping

TRIP_CSV_MAP = build_trip_csv_map()

# ---------- HELPER: generate Base64-encoded sensor chart ----------

PLOT_WINDOW_S = 1.5  # seconds before and after the event centre

def generate_event_chart_b64(raw_df, event_time_s):
    """
    Generate a dual-axis chart (Accelerometer + Gyroscope) around `event_time_s`
    and return the image as a Base64-encoded PNG string for HTML embedding.
    """
    times = raw_df["timestamp"].astype(float) / 1000.0

    t_start = event_time_s - PLOT_WINDOW_S
    t_end   = event_time_s + PLOT_WINDOW_S
    seg     = raw_df[(times >= t_start) & (times <= t_end)]

    if len(seg) < 3:
        return None

    t_rel = seg["timestamp"].astype(float) / 1000.0 - event_time_s  # relative to event centre
    mags  = seg["magnitude"].astype(float).values

    has_gyro = all(c in seg.columns for c in ("gx", "gy", "gz"))

    fig, axes = plt.subplots(
        2 if has_gyro else 1, 1,
        figsize=(4, 3 if has_gyro else 1.8),
        dpi=100,
        sharex=True,
    )

    if not has_gyro:
        axes = [axes]

    # --- Accelerometer subplot ---
    ax_a = axes[0]
    ax_a.plot(t_rel, mags, color="#2563eb", linewidth=0.8)
    ax_a.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.set_ylabel("m/s²", fontsize=7)
    ax_a.set_title("Accelerometer", fontsize=8, pad=2)
    ax_a.tick_params(labelsize=6)
    ax_a.grid(True, alpha=0.3)

    # --- Gyroscope subplot ---
    if has_gyro:
        gx = seg["gx"].fillna(0.0).astype(float).values
        gy = seg["gy"].fillna(0.0).astype(float).values
        gz = seg["gz"].fillna(0.0).astype(float).values
        gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

        ax_g = axes[1]
        ax_g.plot(t_rel, gyro_mag, color="#d97706", linewidth=0.8)
        ax_g.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_g.set_ylabel("rad/s", fontsize=7)
        ax_g.set_xlabel("detik dari event", fontsize=7)
        ax_g.set_title("Gyroscope", fontsize=8, pad=2)
        ax_g.tick_params(labelsize=6)
        ax_g.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------- LOAD CANDIDATES ----------

df = pd.read_csv(CANDIDATE_PATH)
df["datetime"]     = pd.to_datetime(df["time_s"], unit="s")
df["datetime_wib"] = df["datetime"] + pd.Timedelta(hours=7)

trips = df["trip_id"].unique()

# Cek apakah kolom scoring tersedia (dari labeling.py versi baru)
_HAS_SCORE = "score" in df.columns and "priority" in df.columns

print(f"Total Perjalanan (Trips)  : {len(trips)}")
print(f"Total Kandidat Guncangan  : {len(df)}")
print("\nDaftar Trip:")
for i, t in enumerate(trips):
    t_df = df[df["trip_id"] == t]
    if _HAS_SCORE:
        pri = t_df["priority"].value_counts()
        h, m_, lo = int(pri.get("high", 0)), int(pri.get("medium", 0)), int(pri.get("low", 0))
        print(f"  [{i}] {t}  ({len(t_df)} events | H:{h} M:{m_} L:{lo})")
    else:
        print(f"  [{i}] {t}  ({len(t_df)} events)")

# %% [markdown]
# ## 1. Pilih Trip yang Ingin Dilabeli
# Ubah angka `PILIHAN_INDEX_TRIP` untuk memilih rute.

# %%
PILIHAN_INDEX_TRIP = 0   # <<< UBAH ANGKA INI

selected_trip = trips[PILIHAN_INDEX_TRIP]
df_trip       = df[df["trip_id"] == selected_trip].copy()
df_trip       = df_trip.sort_values(by="time_s")

start_time          = df_trip["time_s"].min()
df_trip["detik_ke"] = df_trip["time_s"] - start_time
df_trip["nomor_event"] = range(1, len(df_trip) + 1)

print(f"=== Trip: {selected_trip} ===")
print(f"Jumlah Event : {len(df_trip)}")
print(f"Mulai        : {df_trip['datetime_wib'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')} WIB")
if _HAS_SCORE:
    pri = df_trip["priority"].value_counts()
    print(f"Priority     : High={int(pri.get('high',0))}  Med={int(pri.get('medium',0))}  Low={int(pri.get('low',0))}")
    print(f"Score range  : {df_trip['score'].min():.3f} – {df_trip['score'].max():.3f}")

# Muat data mentah untuk trip ini (dibutuhkan untuk grafik)
raw_csv_path = TRIP_CSV_MAP.get(selected_trip)
df_raw = None
if raw_csv_path:
    print(f"Memuat data mentah: {os.path.basename(raw_csv_path)} ...")
    df_raw = pd.read_csv(raw_csv_path)
    if "magnitude" not in df_raw.columns:
        df_raw["magnitude"] = np.sqrt(df_raw["ax"] ** 2 + df_raw["ay"] ** 2 + df_raw["az"] ** 2)
    print(f"  {len(df_raw)} sampel mentah dimuat.")
else:
    print("[WARN] File CSV mentah tidak ditemukan untuk trip ini. Grafik tidak akan tampil.")

_preview_cols = ["nomor_event", "datetime_wib", "detik_ke", "peak_mag_g", "peak_gyro_mag", "level"]
if _HAS_SCORE:
    _preview_cols += ["score", "priority"]
df_trip.head(5)[_preview_cols]

# %% [markdown]
# ## 2. Peta Interaktif dengan Grafik Sensor
# Setiap popup marker menampilkan:
# - Nomor event & timestamp audio-sync
# - Grafik Accelerometer (m/s²) ± 1.5 detik dari pusat event
# - Grafik Gyroscope (rad/s) di rentang waktu yang sama
# - Garis merah putus-putus menandai titik pusat event

# %%
center_lat = df_trip["lat"].mean()
center_lon = df_trip["lon"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")

# Garis rute
path = list(zip(df_trip["lat"], df_trip["lon"]))
folium.PolyLine(path, color="blue", weight=3, opacity=0.4).add_to(m)

print("Membangkitkan grafik sensor untuk setiap event ...")
chart_count = 0

for _, row in df_trip.iterrows():
    nomor      = int(row["nomor_event"])
    # Tambahkan sedikit simpangan/jitter agar marker tidak saling tumpang tindih sempurna
    lat        = row["lat"] + np.random.uniform(-0.00002, 0.00002)
    lon        = row["lon"] + np.random.uniform(-0.00002, 0.00002)
    waktu_real = row["datetime_wib"].strftime("%H:%M:%S")
    menit      = int(row["detik_ke"] // 60)
    detik      = int(row["detik_ke"] % 60)
    g_force    = row["peak_mag_g"]
    gyro_val   = row.get("peak_gyro_mag", 0)
    level      = row["level"]

    # Ambil kolom scoring (backward-compatible jika belum ada)
    score      = row.get("score", None)
    priority   = row.get("priority", None)
    speed_mean = row.get("speed_mean", None)
    mag_jrk    = row.get("mag_jrk", None)

    # Warna marker: pakai priority jika tersedia, fallback ke level
    if priority:
        color = "red" if priority == "high" else ("orange" if priority == "medium" else "green")
    else:
        color = "red" if level == "high_conf" else ("orange" if level == "candidate" else "green")

    # --- Generate chart ---
    chart_html = ""
    if df_raw is not None:
        b64 = generate_event_chart_b64(df_raw, row["time_s"])
        if b64:
            chart_html = f'<img src="data:image/png;base64,{b64}" style="width:100%;margin-top:6px;">'
            chart_count += 1

    # --- Baris scoring tambahan (hanya jika tersedia) ---
    scoring_rows = ""
    if score is not None:
        speed_kmh = f"{speed_mean * 3.6:.1f} km/h" if speed_mean is not None and speed_mean == speed_mean else "N/A"
        jrk_str   = f"{mag_jrk:.1f}" if mag_jrk is not None and mag_jrk == mag_jrk else "N/A"
        pri_color = "#dc2626" if priority == "high" else ("#d97706" if priority == "medium" else "#16a34a")
        scoring_rows = f"""
          <tr><td>Score</td><td>: <b>{score:.3f}</b></td></tr>
          <tr><td>Priority</td><td>: <b style='color:{pri_color};'>{priority}</b></td></tr>
          <tr><td>Speed</td><td>: {speed_kmh}</td></tr>
          <tr><td>Jerk</td><td>: {jrk_str} m/s²</td></tr>
        """

    popup_html = f"""
    <div style='width:340px; font-family:sans-serif;'>
        <h4 style='margin:0 0 4px 0;'>Event #{nomor}</h4>
        <hr style='margin:0 0 6px 0;'>
        <b style='color:#2563eb;'>🎧 Audio: Menit {menit:02d} Detik {detik:02d}</b><br>
        <span style='font-size:11px;color:#666;'>Waktu: {waktu_real} WIB</span><br>
        <table style='font-size:11px;margin-top:4px;'>
          <tr><td>Accel</td><td>: <b>{g_force:.2f} G</b></td></tr>
          <tr><td>Gyro</td><td>: <b>{gyro_val:.2f} rad/s</b></td></tr>
          <tr><td>Level</td><td>: {level}</td></tr>
          {scoring_rows}
        </table>
        {chart_html}
    </div>
    """

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=380),
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

print(f"Selesai. {chart_count}/{len(df_trip)} event memiliki grafik sensor.")
m

# %% [markdown]
# ## 3. Form Input Label (Ground Truth)
# Cocokkan nomor event di peta dengan rekaman audio Anda, lalu isi label di bawah.
# 
# Label yang disarankan (konsisten bahasa Inggris):
# - `"Normal"` — getaran wajar, bukan kerusakan
# - `"Pothole"` — lubang jalan
# - `"Speed Bump"` — polisi tidur
# - `"Crack"` — retakan/sambungan aspal
# - `"Severe Anomaly"` — kerusakan parah / tidak terklasifikasi

# %%
USER_LABELS = {
    2: "Normal",
    3: "Normal",
    4: "Normal",
    5: "Normal",
    6: "Normal",
    7: "Normal",
    8: "Normal",
    9: "Normal",
    10: "Normal",
    11: "Normal",
    12: "Normal",
    15: "Jembatan",
    23: "Normal",
    28: "Normal",
    30: "Normal",
    34: "Normal",
    38: "Normal",
    40: "Normal",
    43: "Normal",
    44: "Normal",
    45: "Normal",
    46: "Normal",
    47: "Lubang",
    48: "Normal",
    49: "Lubang",
    50: "Lubang",
    51: "Normal",
    52: "Lubang",
    53: "Normal",
    54: "Lubang",
    55: "Lubang",
    56: "Lubang",
    57: "Normal",
    58: "Normal",
    59: "Normal",
    60: "Normal",
    61: "Lubang",
    62: "Lubang", 
    63: "Lubang",
    64: "Lubang",
    65: "Normal",
    66: "Normal",
    67: "Polisi Tidur",
    68: "Jalan Rusak",
    69: "Normal",
    70: "Normal",
    71: "Lubang",
    72: "Jalan Rusak",
    73: "Jalan Rusak",
    75: "Lubang",
    76: "Normal",
    77: "Normal",
    78: "Normal",
    79: "Normal",
    80: "Polisi Tidur",
    82: "Lubang",
    83: "Normal"
    
}

print(f"Total label sesi ini: {len(USER_LABELS)}")

# %% [markdown]
# ## 4. Simpan ke Master Ground Truth
# Label disimpan secara append-safe ke `ground_truth_labels.csv`.
# Event yang di-relabel akan di-overwrite otomatis.

# %%
if len(USER_LABELS) > 0:
    saved_rows = []
    for nomor, label in USER_LABELS.items():
        matched = df_trip[df_trip["nomor_event"] == nomor]
        if not matched.empty:
            saved_rows.append({
                "event_id":   int(matched.iloc[0]["event_id"]),
                "trip_id":    selected_trip,
                "label":      label,
                "labeled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    new_gt_df = pd.DataFrame(saved_rows)

    if os.path.exists(MASTER_GT_PATH):
        master_df = pd.read_csv(MASTER_GT_PATH)
        master_df = master_df[~master_df["event_id"].isin(new_gt_df["event_id"])]
        master_df = pd.concat([master_df, new_gt_df], ignore_index=True)
    else:
        master_df = new_gt_df

    master_df.to_csv(MASTER_GT_PATH, index=False)
    print(f"\n Tersimpan! Total data ground truth: {len(master_df)}")
else:
    print("Belum ada label yang diisi di USER_LABELS.")

# %%
