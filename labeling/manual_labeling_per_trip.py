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

    num_subplots = 6 if has_gyro else 3
    fig, axes = plt.subplots(
        num_subplots, 1,
        figsize=(4.2, 5.5 if has_gyro else 2.8),
        dpi=100,
        sharex=True,
    )

    # --- Accelerometer subplots ---
    if all(c in seg.columns for c in ("ax", "ay", "az")):
        ax_val = seg["ax"].astype(float).values
        ay_val = seg["ay"].astype(float).values
        az_val = seg["az"].astype(float).values
        
        axes[0].plot(t_rel, ax_val, color="#ef4444", linewidth=0.8)
        axes[0].set_ylabel("Acc X", fontsize=6)
        axes[0].set_title("Accelerometer (m/s²)", fontsize=8, pad=2)
        
        axes[1].plot(t_rel, ay_val, color="#22c55e", linewidth=0.8)
        axes[1].set_ylabel("Acc Y", fontsize=6)
        
        axes[2].plot(t_rel, az_val, color="#3b82f6", linewidth=0.8)
        axes[2].set_ylabel("Acc Z", fontsize=6)
        
        for i in range(3):
            axes[i].axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            axes[i].tick_params(labelsize=6)
            axes[i].grid(True, alpha=0.3)
    else:
        axes[0].plot(t_rel, mags, color="#2563eb", linewidth=0.8)
        axes[0].set_ylabel("m/s²", fontsize=6)
        axes[0].set_title("Accelerometer", fontsize=8, pad=2)
        axes[0].axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        axes[0].tick_params(labelsize=6)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_visible(False)
        axes[2].set_visible(False)

    # --- Gyroscope subplots ---
    if has_gyro:
        gx = seg["gx"].fillna(0.0).astype(float).values
        gy = seg["gy"].fillna(0.0).astype(float).values
        gz = seg["gz"].fillna(0.0).astype(float).values

        axes[3].plot(t_rel, gx, color="#ef4444", linewidth=0.8)
        axes[3].set_ylabel("Gyr X", fontsize=6)
        axes[3].set_title("Gyroscope (rad/s)", fontsize=8, pad=2)
        
        axes[4].plot(t_rel, gy, color="#22c55e", linewidth=0.8)
        axes[4].set_ylabel("Gyr Y", fontsize=6)
        
        axes[5].plot(t_rel, gz, color="#3b82f6", linewidth=0.8)
        axes[5].set_ylabel("Gyr Z", fontsize=6)
        axes[5].set_xlabel("detik dari event", fontsize=7)
        
        for i in range(3, 6):
            axes[i].axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
            axes[i].tick_params(labelsize=6)
            axes[i].grid(True, alpha=0.3)
    else:
        axes[2].set_xlabel("detik dari event", fontsize=7)

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
    print(f"  {len(df_raw)} sampel mentah dimuat.")
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

    # --- Fitur Shape Baru ---
    num_peaks_accel = row.get("num_peaks_accel", None)
    num_peaks_gyro  = row.get("num_peaks_gyro", None)
    asym_score      = row.get("asymmetry_score", None)
    loc_dur         = row.get("local_duration", None)

    # Ambil kolom scoring (backward-compatible jika belum ada)
    score      = row.get("score", None)
    priority   = row.get("priority", None)
    speed_mean = row.get("speed_mean", None)
    mag_jrk    = row.get("mag_jrk", None)

    # --- Get Sugesti ---
    suggested_lbl  = row.get("suggested_label", "")
    suggested_conf = row.get("suggestion_confidence", 0.0)
    suggested_rsn  = row.get("suggestion_reason", "")
    suggested_raw  = row.get("suggested_raw_label", "")
    
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
        jrk_str   = f"{mag_jrk:.1f}" if mag_jrk is not None and mag_jrk == mag_jrk else "N/A"
        pri_color = "#dc2626" if priority == "high" else ("#d97706" if priority == "medium" else "#16a34a")
        scoring_rows += f"""
          <tr><td>Score</td><td>: <b>{score:.3f}</b></td></tr>
          <tr><td>Priority</td><td>: <b style='color:{pri_color};'>{priority}</b></td></tr>
          <tr><td>Speed</td><td>: {speed_kmh}</td></tr>
          <tr><td>Jerk</td><td>: {jrk_str} m/s²</td></tr>
        """
        if suggested_lbl:
            scoring_rows += f"""
              <tr><td colspan='2'><hr style='margin:2px 0;'></td></tr>
              <tr><td colspan='2'>🤖 <b>Suggested: <span style='color:#2563eb;'>{suggested_raw}</span></b> ({suggested_conf*100:.0f}%)</td></tr>
              <tr><td colspan='2' style='font-size:10px;color:#555;'><i>{suggested_rsn}</i></td></tr>
            """
        
    if asym_score is not None and asym_score == asym_score: # NaN check
        scoring_rows += f"""
          <tr><td colspan='2'><hr style='margin:2px 0;'></td></tr>
          <tr><td>Peaks (Acc/Gyr)</td><td>: {int(num_peaks_accel)} / <b>{int(num_peaks_gyro)}</b></td></tr>
          <tr><td>Asymmetry</td><td>: <b>{asym_score:.2f}</b></td></tr>
          <tr><td>Duration</td><td>: {loc_dur:.2f}s</td></tr>
        """

    nav_buttons = "<div style='margin-top: 8px; display: flex; justify-content: space-between;'>"
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
        {nav_buttons}
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
# Cocokkan nomor event di peta dengan rekaman audio Anda, lalu isi label di bawah.
# 
# Label yang disarankan (konsisten bahasa Inggris):
# - `"Non-Event"` — semua yang bukan event diskrit, termasuk rough road, engine vibration, maneuver, dan noise.
# - `"Pothole"` — lubang jalan
# - `"Speed Bump"` — polisi tidur
# %%
USER_LABELS = {
    2: "Non-Event",
    3: "Non-Event",
    4: "Non-Event",
    5: "Non-Event",
    6: "Non-Event",
    7: "Non-Event",
    8: "Non-Event",
    9: "Non-Event",
    10: "Non-Event",
    11: "Non-Event",
    12: "Non-Event",
    14: "Pothole",
    15: "Non-Event",
    16: "Non-Event",
    17: "Non-Event",
    23: "Non-Event",
    24: "Non-Event",
    26: "Non-Event",
    28: "Non-Event",
    29: "Non-Event",
    30: "Non-Event",
    31: "Non-Event",
    32: "Non-Event",
    34: "Non-Event",
    38: "Non-Event",
    40: "Non-Event",
    43: "Non-Event",
    44: "Non-Event",
    45: "Non-Event",
    46: "Non-Event",
    47: "Speed Bump",
    48: "Non-Event",
    49: "Non-Event",
    50: "Pothole",
    51: "Non-Event",
    52: "Pothole",
    53: "Non-Event",
    54: "Pothole",
    55: "Pothole",
    56: "Pothole",
    57: "Non-Event",
    58: "Non-Event",
    59: "Non-Event",
    60: "Non-Event",
    61: "Pothole",
    62: "Pothole", 
    63: "Pothole",
    64: "Pothole",
    65: "Non-Event",
    66: "Non-Event",
    67: "Speed Bump",
    68: "Non-Event",
    69: "Non-Event",
    70: "Non-Event",
    71: "Pothole",
    72: "Non-Event",
    73: "Non-Event",
    75: "Pothole",
    76: "Non-Event",
    77: "Non-Event",
    78: "Non-Event",
    79: "Non-Event",
    80: "Speed Bump",
    82: "Pothole",
    83: "Non-Event"
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
