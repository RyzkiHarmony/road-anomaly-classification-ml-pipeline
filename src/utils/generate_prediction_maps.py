import os
import glob
import pandas as pd
import numpy as np
import folium
import json
from scipy.signal import find_peaks
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pengaturan Folder
RAW_DATA_DIR = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\data\raw\road test"
OUT_FOLDER = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\maps_prediction"

os.makedirs(OUT_FOLDER, exist_ok=True)

def generate_event_chart_b64(raw_df, event_time_s):
    PLOT_WINDOW_S = 1.5
    times = raw_df["timestamp"].astype(float) / 1000.0
    t_start = event_time_s - PLOT_WINDOW_S
    t_end   = event_time_s + PLOT_WINDOW_S
    seg     = raw_df[(times >= t_start) & (times <= t_end)]
    
    if len(seg) < 3:
        return None
        
    t_rel = seg["timestamp"].astype(float) / 1000.0 - event_time_s
    has_gyro = all(c in seg.columns for c in ("gx", "gy", "gz"))
    
    num_subplots = 3 + (3 if has_gyro else 0)
    fig_height = 1.3 * num_subplots
    fig, axes = plt.subplots(num_subplots, 1, figsize=(4.2, fig_height), dpi=100, sharex=True)
    if num_subplots == 1:
        axes = [axes]
    
    ax_idx = 0
    if all(c in seg.columns for c in ("ax", "ay", "az")):
        axes[ax_idx].plot(t_rel, seg["ax"].astype(float), color="#eab308", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc X", fontsize=6)
        axes[ax_idx].set_title("Accelerometer raw (m/s²)", fontsize=8, pad=2)
        ax_idx += 1
        
        axes[ax_idx].plot(t_rel, seg["ay"].astype(float), color="#22c55e", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc Y", fontsize=6)
        ax_idx += 1
        
        axes[ax_idx].plot(t_rel, seg["az"].astype(float), color="#3b82f6", linewidth=0.8)
        axes[ax_idx].set_ylabel("Acc Z", fontsize=6)
        ax_idx += 1
        
    if has_gyro:
        axes[ax_idx].plot(t_rel, seg["gx"].astype(float), color="#eab308", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr X", fontsize=6)
        axes[ax_idx].set_title("Gyroscope (rad/s)", fontsize=8, pad=2)
        ax_idx += 1
        
        axes[ax_idx].plot(t_rel, seg["gy"].astype(float), color="#22c55e", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr Y", fontsize=6)
        ax_idx += 1
        
        axes[ax_idx].plot(t_rel, seg["gz"].astype(float), color="#3b82f6", linewidth=0.8)
        axes[ax_idx].set_ylabel("Gyr Z", fontsize=6)
        ax_idx += 1
        
    for ax in axes:
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

def extract_events(df, prob_col, event_type, threshold=0.35, cooldown_ms=2000):
    """
    Ekstrak event menggunakan logika 'First-Trigger Cooldown' (seperti di Android).
    Menangkap probabilitas saat pertama kali menyentuh threshold, lalu mengabaikan 
    data berikutnya selama masa cooldown (default 2 detik).
    """
    probs = df[prob_col].fillna(0).values
    timestamps = df['timestamp'].values
    
    events = []
    last_trigger_time = -9999999
    
    for i in range(len(probs)):
        prob = float(probs[i])
        now = timestamps[i]
        
        if prob >= threshold:
            if (now - last_trigger_time) > cooldown_ms:
                last_trigger_time = now
                events.append({
                    'lat': float(df.iloc[i]['lat']),
                    'lon': float(df.iloc[i]['lon']),
                    'prob': prob,
                    'type': event_type,
                    'time': int(now),
                    'prob_col': prob_col
                })
                
    return events

def create_map(trip_id, df, events, out_path, csv_path):
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")
    
    # Gambar garis rute perjalanan
    path_coords = df[['lat', 'lon']].dropna().values.tolist()
    folium.PolyLine(path_coords, color="blue", weight=3, opacity=0.4).add_to(m)
    
    # Cari file JSON untuk trip ini agar detik audio sinkron dengan startTime
    json_path = csv_path.replace('.csv', '.json')
    meta = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
    csv_start_s = df['timestamp'].min() / 1000.0
    trip_start_s = csv_start_s
    
    if meta and "startTime" in meta:
        meta_start_s = meta["startTime"] / 1000.0
        # Cek apakah timebase-nya sama (jika CSV pakai uptimeMillis dan JSON pakai Epoch, akan selisih besar)
        if abs(csv_start_s - meta_start_s) < 86400 * 365: # Jika selisih < 1 tahun, berarti timebase sama
            trip_start_s = meta_start_s
            print(f"    [INFO] Menggunakan startTime dari JSON untuk sinkronisasi detik audio.")
        else:
            print(f"    [WARN] Timebase CSV dan JSON berbeda. Menggunakan event pertama sebagai Detik 0.")
    else:
        print(f"    [WARN] Metadata JSON tidak ditemukan. Menggunakan event pertama sebagai Detik 0.")
    
    events = sorted(events, key=lambda x: x['time'])
    events_metadata = []
    
    for i, e in enumerate(events):
        # Tambahkan sedikit jitter (simpangan) agar jika ada event berhimpitan tidak saling menutupi sempurna
        lat = e['lat'] + np.random.uniform(-0.00002, 0.00002)
        lon = e['lon'] + np.random.uniform(-0.00002, 0.00002)
        
        event_time_s = e['time'] / 1000.0
        detik_ke = event_time_s - trip_start_s
        menit = int(detik_ke // 60)
        detik = int(detik_ke % 60)
        
        b64 = generate_event_chart_b64(df, event_time_s)
        chart_html = f'<img src="data:image/png;base64,{b64}" style="width:100%;margin-top:6px;">' if b64 else ""
        
        color = 'red' if e['type'] == 'Pothole' else 'orange'
        icon_type = 'exclamation-sign' if e['type'] == 'Pothole' else 'road'
        
        nav_buttons = "<div style='margin-bottom: 8px; display: flex; justify-content: space-between;'>"
        if i > 0:
            nav_buttons += f"<button onclick='goToEvent({i-1})' style='cursor:pointer; padding:2px 8px; font-size:11px;'>&laquo; Prev</button>"
        else:
            nav_buttons += "<div></div>"
            
        if i < len(events) - 1:
            nav_buttons += f"<button onclick='goToEvent({i+1})' style='cursor:pointer; padding:2px 8px; font-size:11px;'>Next &raquo;</button>"
        else:
            nav_buttons += "<div></div>"
        nav_buttons += "</div>"
        
        popup_html = f"""
        <div style='font-family:sans-serif; width: 340px;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h4 style='margin:0 0 4px 0; color:{color};'>{e['type']}</h4>
            </div>
            {nav_buttons}
            <hr style='margin:0 0 6px 0;'>
            <b style='color:#2563eb;'>🎧 Audio: Menit {menit:02d} Detik {detik:02d}</b><br>
            <b>Probabilitas:</b> {e['prob']*100:.1f}%<br>
            <span style='font-size:11px;color:#666;'>Lat: {lat:.5f} | Lon: {lon:.5f}</span><br>
            {chart_html}
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=380),
            icon=folium.Icon(color=color, icon=icon_type)
        ).add_to(m)
        
        events_metadata.append({
            'index': i,
            'lat': lat,
            'lon': lon,
            'prob': e['prob']
        })
        
    # --- Inject HTML Tombol Filter ---
    buttons_html = """
    <div style="position: fixed; top: 15px; left: 50px; z-index: 9999; background: white; padding: 12px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-family: sans-serif;">
        <h4 style="margin: 0 0 10px 0; font-size: 14px;">Filter Probabilitas Event</h4>
        <button onclick="filterMarkers(0.35, this)" class="btn-filter" style="cursor:pointer; padding:6px 12px; margin-right:5px; background:#2563eb; color:white; border:none; border-radius:4px; font-weight:bold;">&ge; 35%</button>
        <button onclick="filterMarkers(0.50, this)" class="btn-filter" style="cursor:pointer; padding:6px 12px; margin-right:5px; background:#64748b; color:white; border:none; border-radius:4px;">&ge; 50%</button>
        <button onclick="filterMarkers(0.75, this)" class="btn-filter" style="cursor:pointer; padding:6px 12px; margin-right:5px; background:#64748b; color:white; border:none; border-radius:4px;">&ge; 75%</button>
    </div>
    """
    m.get_root().html.add_child(folium.Element(buttons_html))
    
    # --- Inject JavaScript Logic ---
    events_json = json.dumps(events_metadata)
    js_script = f"""
    <script>
    var eventsMeta = {events_json};
    var allMarkers = [];
    var mapInstance = null;
    
    document.addEventListener("DOMContentLoaded", function() {{
        // Cari instance map L.Map di object window global
        for (var key in window) {{
            if (window[key] instanceof L.Map) {{
                mapInstance = window[key];
                break;
            }}
        }}
        
        if (mapInstance) {{
            mapInstance.eachLayer(function(layer) {{
                if (layer instanceof L.Marker) {{
                    var mLat = layer.getLatLng().lat;
                    var mLon = layer.getLatLng().lng;
                    
                    // Cocokkan marker Leaflet dengan metadata Python via koordinat
                    var match = eventsMeta.find(function(em) {{
                        return Math.abs(em.lat - mLat) < 1e-6 && Math.abs(em.lon - mLon) < 1e-6;
                    }});
                    
                    if (match) {{
                        layer.prob = match.prob; // Attach probability custom
                        allMarkers.push(layer);
                    }}
                }}
            }});
        }}
    }});
    
    window.filterMarkers = function(minProb, btnElement) {{
        if (!mapInstance) return;
        
        // Update warna tombol aktif
        var buttons = document.getElementsByClassName("btn-filter");
        for(var i=0; i<buttons.length; i++) {{
            buttons[i].style.backgroundColor = "#64748b";
            buttons[i].style.fontWeight = "normal";
        }}
        btnElement.style.backgroundColor = "#2563eb";
        btnElement.style.fontWeight = "bold";
        
        var count = 0;
        
        allMarkers.forEach(function(m) {{
            if (m.prob >= minProb) {{
                if (!mapInstance.hasLayer(m)) mapInstance.addLayer(m);
                count++;
            }} else {{
                if (mapInstance.hasLayer(m)) mapInstance.removeLayer(m);
            }}
        }});
        
        console.log("Filtered events (>= " + minProb*100 + "%): " + count);
    }};
    
    window.goToEvent = function(index) {{
        if (!mapInstance) return;
        var ev = eventsMeta[index];
        if (!ev) return;
        
        mapInstance.setView([ev.lat, ev.lon], 18);
        mapInstance.eachLayer(function(layer) {{
            if (layer instanceof L.Marker) {{
                var mLat = layer.getLatLng().lat;
                var mLon = layer.getLatLng().lng;
                if (Math.abs(mLat - ev.lat) < 1e-6 && Math.abs(mLon - ev.lon) < 1e-6) {{
                    layer.openPopup();
                }}
            }}
        }});
    }};
    </script>
    """
    m.get_root().html.add_child(folium.Element(js_script))
    
    m.save(out_path)
    print(f"  -> Tersimpan: {out_path} ({len(events)} event terdeteksi)")


def main():
    print(f"Mencari data trip di: {RAW_DATA_DIR}")
    trip_folders = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "Trip_*")))
    
    if not trip_folders:
        print("Tidak ada folder trip yang ditemukan.")
        return
        
    for trip_dir in trip_folders:
        trip_id = os.path.basename(trip_dir)
        csv_files = glob.glob(os.path.join(trip_dir, "*.csv"))
        if not csv_files:
            continue
            
        csv_path = csv_files[0]
        print(f"\nMemproses {trip_id} ...")
        
        df = pd.read_csv(csv_path)
        
        # Validasi kolom
        req_cols = ['lat', 'lon', 'prob_pothole', 'prob_speedbump']
        if not all(c in df.columns for c in req_cols):
            print(f"  [WARN] Kolom yang dibutuhkan tidak lengkap di {trip_id}. Melewati trip ini.")
            continue
            
        # Buang baris tanpa koordinat GPS
        df = df.dropna(subset=['lat', 'lon'])
        if len(df) == 0:
            print("  [WARN] Data kosong setelah filter GPS.")
            continue
            
        # Ekstrak puncak probabilitas (events) dengan batas minimum 35%
        potholes = extract_events(df, 'prob_pothole', 'Pothole', threshold=0.35)
        bumps = extract_events(df, 'prob_speedbump', 'Speed Bump', threshold=0.35)
        
        all_events = potholes + bumps
        
        if len(all_events) == 0:
            print("  [INFO] Tidak ada event >= 35% pada trip ini.")
            
        out_path = os.path.join(OUT_FOLDER, f"map_{trip_id}.html")
        
        # Tambahkan passing argumen csv_path untuk create_map agar bisa membaca JSON
        create_map(trip_id, df, all_events, out_path, csv_path)
        
    print("\nSelesai! Semua peta interaktif disimpan di:")
    print(OUT_FOLDER)

if __name__ == "__main__":
    main()
