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
from label_suggester import apply_label_suggestions, save_label_suggestions
from sensor_fusion import apply_sensor_fusion

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
    Generate a multi-panel chart around `event_time_s`:
      Panel 1-3 : Raw Accelerometer X / Y / Z
      Panel 4   : Vertical Linear Acceleration (a_vertical) – gravity removed & aligned
      Panel 5-7 : Gyroscope X / Y / Z  (if available)
    Returns the image as a Base64-encoded PNG string for HTML embedding.
    """
    times = raw_df["timestamp"].astype(float) / 1000.0

    t_start = event_time_s - PLOT_WINDOW_S
    t_end   = event_time_s + PLOT_WINDOW_S
    seg     = raw_df[(times >= t_start) & (times <= t_end)]

    if len(seg) < 3:
        return None

    t_rel  = seg["timestamp"].astype(float) / 1000.0 - event_time_s
    has_gyro    = all(c in seg.columns for c in ("gx", "gy", "gz"))
    has_vertical = "a_vertical" in seg.columns

    # Determine number of subplots: Acc(3) + Vert(1) + Gyro(3 if available)
    num_subplots = 3 + (1 if has_vertical else 0) + (3 if has_gyro else 0)
    fig_height   = 1.3 * num_subplots
    fig, axes = plt.subplots(
        num_subplots, 1,
        figsize=(4.2, fig_height),
        dpi=100,
        sharex=True,
    )
    if num_subplots == 1:
        axes = [axes]

    ax_idx = 0

    # --- Accelerometer subplots (raw X/Y/Z) ---
    if all(c in seg.columns for c in ("ax", "ay", "az")):
        ax_val = seg["ax"].astype(float).values
        ay_val = seg["ay"].astype(float).values
        az_val = seg["az"].astype(float).values

        axes[ax_idx].plot(t_rel, ax_val, color="#eab308", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc X", fontsize=6)
        axes[ax_idx].set_title("Accelerometer raw (m/s²)", fontsize=8, pad=2)
        ax_idx += 1

        axes[ax_idx].plot(t_rel, ay_val, color="#22c55e", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc Y", fontsize=6)
        ax_idx += 1

        axes[ax_idx].plot(t_rel, az_val, color="#3b82f6", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc Z", fontsize=6)
        ax_idx += 1
    else:
        mags = seg["magnitude"].astype(float).values
        axes[ax_idx].plot(t_rel, mags, color="#2563eb", linewidth=0.8)
        axes[ax_idx].set_ylabel("Mag", fontsize=6)
        axes[ax_idx].set_title("Accelerometer (m/s²)", fontsize=8, pad=2)
        ax_idx += 1

    # --- Vertical acceleration subplot ---
    if has_vertical:
        a_vert = seg["a_vertical"].astype(float).values
        axes[ax_idx].plot(t_rel, a_vert, color="#a855f7", linewidth=1.0)
        axes[ax_idx].axhline(0, color="gray", linewidth=0.5, linestyle=":")
        axes[ax_idx].set_ylabel("a_vert", fontsize=6)
        axes[ax_idx].set_title("Vertical Accel (m/s², +up/-down)", fontsize=8, pad=2)
        ax_idx += 1

    # --- Gyroscope subplots ---
    if has_gyro:
        gx = seg["gx"].fillna(0.0).astype(float).values
        gy = seg["gy"].fillna(0.0).astype(float).values
        gz = seg["gz"].fillna(0.0).astype(float).values

        axes[ax_idx].plot(t_rel, gx, color="#eab308", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr X", fontsize=6)
        axes[ax_idx].set_title("Gyroscope (rad/s)", fontsize=8, pad=2)
        ax_idx += 1

        axes[ax_idx].plot(t_rel, gy, color="#22c55e", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr Y", fontsize=6)
        ax_idx += 1

        axes[ax_idx].plot(t_rel, gz, color="#3b82f6", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr Z", fontsize=6)
        ax_idx += 1

    # Shared formatting
    for i, ax in enumerate(axes):
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("detik dari event", fontsize=7)

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
PILIHAN_INDEX_TRIP = 0   # <<< PILIH TRIP / GANTI DATA

selected_trip = trips[PILIHAN_INDEX_TRIP]
df_trip       = df[df["trip_id"] == selected_trip].copy()
df_trip       = df_trip.sort_values(by="time_s")

if "suggested_label" not in df_trip.columns:
    df_trip = apply_label_suggestions(df_trip)

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
    # Apply sensor fusion to add a_vertical column for chart visualisation
    try:
        df_raw = apply_sensor_fusion(df_raw)
        print(f"  {len(df_raw)} sampel mentah dimuat (sensor fusion applied).")
    except ValueError as e:
        print(f"[ERROR] Failed to apply sensor fusion on raw data: {e}")
        df_raw = None
else:
    print("[WARN] File CSV mentah tidak ditemukan untuk trip ini. Grafik tidak akan tampil.")

_preview_cols = ["nomor_event", "datetime_wib", "detik_ke", "peak_mag_g", "peak_gyro_mag", "level"]
if _HAS_SCORE:
    _preview_cols += ["score", "priority"]
if "asymmetry_score" in df_trip.columns:
    _preview_cols += ["num_peaks_gyro", "asymmetry_score"]
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

event_coords = {}
nomors = df_trip["nomor_event"].tolist()

for i, (_, row) in enumerate(df_trip.iterrows()):
    nomor      = int(row["nomor_event"])
    prev_nomor = nomors[i-1] if i > 0 else None
    next_nomor = nomors[i+1] if i < len(nomors) - 1 else None
    # Tambahkan sedikit simpangan/jitter agar marker tidak saling tumpang tindih sempurna
    lat        = row["lat"] + np.random.uniform(-0.00002, 0.00002)
    lon        = row["lon"] + np.random.uniform(-0.00002, 0.00002)
    
    event_coords[nomor] = {"lat": lat, "lon": lon}
    waktu_real = row["datetime_wib"].strftime("%H:%M:%S")
    menit      = int(row["detik_ke"] // 60)
    detik      = int(row["detik_ke"] % 60)
    g_force    = row["peak_mag_g"]
    gyro_val   = row.get("peak_gyro_mag", 0)
    level      = row["level"]

    # --- Fitur Shape & Domain Baru (S-Tier) ---
    num_peaks_accel = row.get("num_peaks_accel", None)
    num_peaks_gyro  = row.get("num_peaks_gyro", None)
    kurtosis_val    = row.get("kurtosis", None)
    max_jerk_val    = row.get("max_jerk", None)
    duration_thr    = row.get("duration_above_threshold", None)
    fft_ratio       = row.get("fft_high_low_ratio", None)

    # Ambil kolom scoring (backward-compatible jika belum ada)
    score      = row.get("score", None)
    priority   = row.get("priority", None)
    speed_mean = row.get("speed_mean", None)

    # --- Get Sugesti ---
    suggested_lbl  = row.get("suggested_label", "")
    suggested_conf = row.get("suggestion_confidence", 0.0)
    suggested_rsn  = row.get("suggestion_reason", "")
    suggested_raw  = row.get("suggested_raw_label", "")
    
    vert_val = row.get('peak_vertical_g', float('nan'))
    vert_dir_str = "<span style='color:#ef4444;'>⬇️ Down (Pothole?)</span>" if vert_val < 0 else "<span style='color:#16a34a;'>⬆️ Up (Bump?)</span>"

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

    # --- Baris scoring & shape tambahan (hanya jika tersedia) ---
    scoring_rows = ""
    if score is not None:
        speed_kmh = f"{speed_mean * 3.6:.1f} km/h" if speed_mean is not None and speed_mean == speed_mean else "N/A"
        pri_color = "#dc2626" if priority == "high" else ("#d97706" if priority == "medium" else "#16a34a")
        scoring_rows += f"""
          <tr><td>Priority / Score</td><td>: <b style='color:{pri_color};'>{priority}</b> ({score:.3f})</td></tr>
          <tr><td>Speed</td><td>: {speed_kmh}</td></tr>
        """
        if suggested_lbl:
            scoring_rows += f"""
              <tr><td colspan='2'><hr style='margin:2px 0;'></td></tr>
              <tr><td colspan='2'>🤖 <b>Suggested: <span style='color:#8b5cf6;'>{suggested_lbl}</span></b> ({suggested_conf:.0%})</td></tr>
              <tr><td colspan='2' style='font-size:10px;color:#555;'><i>{suggested_rsn}</i></td></tr>
            """
        
    if kurtosis_val is not None and kurtosis_val == kurtosis_val: # NaN check
        jrk_str = f"{max_jerk_val:.0f}" if max_jerk_val is not None else "N/A"
        kur_str = f"{kurtosis_val:.1f}" if kurtosis_val is not None else "N/A"
        fft_str = f"{fft_ratio:.1f}" if fft_ratio is not None else "N/A"
        
        scoring_rows += f"""
          <tr><td colspan='2'><hr style='margin:2px 0;'></td></tr>
          <tr><td>Max Jerk</td><td>: <b>{jrk_str} m/s³</b> <span style='font-size:9px;color:#888'>(Tinggi=Pothole)</span></td></tr>
          <tr><td>Kurtosis</td><td>: <b>{kur_str}</b> <span style='font-size:9px;color:#888'>(Tinggi=Pothole)</span></td></tr>
          <tr><td>Duration</td><td>: <b>{duration_thr:.2f}s</b> <span style='font-size:9px;color:#888'>(Panjang=Bump)</span></td></tr>
          <tr><td>FFT (Hi/Lo)</td><td>: {fft_str}</td></tr>
          <tr><td>Peaks (Acc/Gyr)</td><td>: {int(num_peaks_accel)} / {int(num_peaks_gyro)}</td></tr>
        """

    nav_buttons = "<div style='margin-bottom: 8px; display: flex; justify-content: space-between;'>"
    if prev_nomor:
        nav_buttons += f"<button onclick='goToEvent({prev_nomor})' style='cursor:pointer; padding:2px 8px; font-size:11px;'>&laquo; Prev</button>"
    else:
        nav_buttons += "<div></div>"
        
    if next_nomor:
        nav_buttons += f"<button onclick='goToEvent({next_nomor})' style='cursor:pointer; padding:2px 8px; font-size:11px;'>Next &raquo;</button>"
    else:
        nav_buttons += "<div></div>"
    nav_buttons += "</div>"

    gmaps_url = f"https://www.google.com/maps?q={lat},{lon}"

    popup_html = f"""
    <div style='width:340px; font-family:sans-serif;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h4 style='margin:0 0 4px 0;'>Event #{nomor}</h4>
            <button onclick="copyGmaps('{gmaps_url}', this)" style='cursor:pointer; font-size:11px; color:#ffffff; background-color:#16a34a; padding:2px 6px; border:none; border-radius:3px;'>📍 Salin Maps URL</button>
        </div>
        {nav_buttons}
        <hr style='margin:0 0 6px 0;'>
        <b style='color:#2563eb;'>🎧 Audio: Menit {menit:02d} Detik {detik:02d}</b><br>
        <span style='font-size:11px;color:#666;'>Waktu: {waktu_real} WIB</span><br>
        <table style='font-size:11px;margin-top:4px;'>
          <tr><td>Vert (peak)</td><td>: <b>{vert_val:.2f} G</b> {vert_dir_str}</td></tr>
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

js_script = """
<script>
var eventCoords = {
"""
for k, v in event_coords.items():
    js_script += f"    {k}: [{v['lat']}, {v['lon']}],\n"
js_script += """};
function goToEvent(nomor) {
    var coords = eventCoords[nomor];
    if (!coords) return;
    
    var mapInstance = null;
    for (var key in window) {
        if (window[key] instanceof L.Map) {
            mapInstance = window[key];
            break;
        }
    }
    if (!mapInstance) return;
    
    mapInstance.setView(coords, 18);
    
    mapInstance.eachLayer(function(layer) {
        if (layer instanceof L.Marker) {
            var mLat = layer.getLatLng().lat;
            var mLon = layer.getLatLng().lng;
            if (Math.abs(mLat - coords[0]) < 1e-6 && Math.abs(mLon - coords[1]) < 1e-6) {
                layer.openPopup();
            }
        }
    });
}

function copyGmaps(url, btn) {
    var temp = document.createElement("textarea");
    temp.value = url;
    document.body.appendChild(temp);
    temp.select();
    try {
        document.execCommand("copy");
        btn.innerHTML = "✅ Tersalin!";
        btn.style.backgroundColor = "#2563eb";
        setTimeout(function() {
            btn.innerHTML = "📍 Salin Maps URL";
            btn.style.backgroundColor = "#16a34a";
        }, 2000);
    } catch (err) {
        alert("Gagal menyalin URL.");
    }
    document.body.removeChild(temp);
}
</script>
"""
m.get_root().html.add_child(folium.Element(js_script))

print(f"Selesai. {chart_count}/{len(df_trip)} event memiliki grafik sensor.")
m

# %% [markdown]
# ## 3. Form Input Label (Ground Truth)
# 
# ### Cara kerja:
# 1. Jalankan sel di bawah untuk memuat label dari file JSON
#    (jika file belum ada, akan dibuat template kosong).
# 2. Edit file JSON di `labels/<trip_id>_labels.json`.
# 3. Jalankan sel ini lagi untuk memuat perubahan.
# 4. Atau, edit dict `USER_LABELS` langsung di bawah lalu jalankan sel berikutnya.
#
# Label yang tersedia (konsisten bahasa Inggris):
# - `"Non-Event"` — semua yang bukan event diskrit (rough road, engine vibration, maneuver, noise)
# - `"Pothole"` — lubang jalan
# - `"Speed Bump"` — polisi tidur

# %%
LABELS_FOLDER = os.path.join(_DIR, "labels")
os.makedirs(LABELS_FOLDER, exist_ok=True)


def _label_file_path(trip_id):
    """Generate path file label JSON berdasarkan trip_id."""
    # Gunakan 8 karakter pertama UUID agar nama file tidak terlalu panjang
    short_id = str(trip_id).split("-")[0] if "-" in str(trip_id) else str(trip_id)
    return os.path.join(LABELS_FOLDER, f"{short_id}_labels.json")


def load_labels_from_json(trip_id, trip_index=None):
    """
    Load USER_LABELS dari file JSON.
    Jika file belum ada, buat template kosong dan kembalikan dict kosong.
    """
    path = _label_file_path(trip_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Konversi key string → int
        labels = {int(k): v for k, v in data.get("labels", {}).items()}
        print(f"[OK] Dimuat {len(labels)} label dari: {os.path.basename(path)}")
        return labels

    # Buat template kosong
    template = {
        "trip_index": trip_index if trip_index is not None else "?",
        "trip_id": str(trip_id),
        "labeled_at": "",
        "notes": "Edit labels dict di bawah, lalu jalankan sel berikutnya.",
        "labels": {}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=4, ensure_ascii=False)
    print(f"[INFO] File label kosong dibuat: {os.path.basename(path)}")
    print(f"       Edit file tersebut, lalu jalankan sel ini lagi.")
    return {}


def save_labels_to_json(trip_id, labels, trip_index=None, notes=""):
    """Simpan USER_LABELS ke file JSON untuk referensi masa depan."""
    path = _label_file_path(trip_id)
    data = {
        "trip_index": trip_index if trip_index is not None else "?",
        "trip_id": str(trip_id),
        "labeled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "labels": {str(k): v for k, v in sorted(labels.items())}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[OK] {len(labels)} label disimpan ke: {os.path.basename(path)}")


# Muat label dari file JSON (atau buat template kosong)
USER_LABELS = load_labels_from_json(selected_trip, PILIHAN_INDEX_TRIP)

# ──────────────────────────────────────────────────────────────────────
# Jika lebih suka mengedit langsung di notebook, uncomment dan isi di sini:
#
# USER_LABELS = {
#     1: "Non-Event",
#     2: "Pothole",
#     3: "Speed Bump",
#     # ... tambahkan sesuai kebutuhan
# }
# ──────────────────────────────────────────────────────────────────────

print(f"Total label sesi ini: {len(USER_LABELS)}")

# %% [markdown]
# ## 4. Simpan ke Master Ground Truth
# Label disimpan secara append-safe ke `ground_truth_labels.csv`.
# Event yang di-relabel akan di-overwrite otomatis.
# File JSON juga diperbarui sebagai backup.

# %%
if len(USER_LABELS) > 0:
    saved_rows = []
    for nomor, label in USER_LABELS.items():
        matched = df_trip[df_trip["nomor_event"] == nomor]
        if not matched.empty:
            dur = float(matched.iloc[0].get("duration_above_threshold", 0.2))
            if pd.isna(dur) or dur <= 0: dur = 0.2
            t_peak = float(matched.iloc[0]["time_s"])
            
            saved_rows.append({
                "event_id":         int(matched.iloc[0]["event_id"]),
                "trip_id":          selected_trip,
                "label":            label,
                "event_start":      t_peak - (dur / 2.0),
                "event_peak":       t_peak,
                "event_end":        t_peak + (dur / 2.0),
                "confidence":       1.0,
                "annotator":        "manual",
                "notes":            "",
                "created_at":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": "01_labeling_pipeline"
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

    # Simpan juga ke file JSON sebagai backup
    save_labels_to_json(selected_trip, USER_LABELS, PILIHAN_INDEX_TRIP)
else:
    print("Belum ada label yang diisi di USER_LABELS.")
    print(f"Edit file: {_label_file_path(selected_trip)}")
    print("Lalu jalankan sel sebelumnya untuk memuat label.")

# %%
