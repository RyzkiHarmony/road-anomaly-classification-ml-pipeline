import os
import glob
import json
import pandas as pd
import folium
import sys

# Tambahkan path utilitas agar bisa import config
_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(os.path.join(_DIR, '..', 'src', 'utils'))
from config import CSV_FOLDER, META_FOLDER, OUT_FOLDER

MAP_FOLDER = os.path.join(OUT_FOLDER, "maps")
os.makedirs(MAP_FOLDER, exist_ok=True)

COLORS = [
    '#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#264653',
    '#8AB17D', '#B5838D', '#FFB4A2', '#6D597A', '#355070',
    '#118AB2', '#06D6A0', '#FFD166', '#EF476F', '#A05195',
    '#D45087', '#5E60CE', '#4EA8DE', '#5390D9', '#48BFE3'
]

def build_trip_csv_map():
    """Memetakan trip_id ke path CSV raw berdasarkan file metadata JSON."""
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

def main():
    trip_csv_map = build_trip_csv_map()
    if not trip_csv_map:
        print("Tidak ada data trip yang ditemukan.")
        return

    # Inisialisasi map dengan CartoDB positron (OSM Carto Style Light)
    m = folium.Map(location=[-7.25, 112.75], zoom_start=12, tiles="CartoDB positron")
    
    all_lat = []
    all_lon = []
    
    print(f"Ditemukan {len(trip_csv_map)} trip, membuat peta rute...")
    
    for i, (trip_id, csv_path) in enumerate(trip_csv_map.items()):
        print(f"Memproses {trip_id}...")
        df = pd.read_csv(csv_path)
        
        # Buang baris tanpa data koordinat
        df = df.dropna(subset=['lat', 'lon'])
        
        # Ekstrak data koordinat
        coords = list(zip(df['lat'], df['lon']))
        if not coords:
            continue
            
        all_lat.extend(df['lat'])
        all_lon.extend(df['lon'])
            
        color = COLORS[i % len(COLORS)]
        
        # Gambar rute menggunakan PolyLine
        folium.PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.8,
            tooltip=f"Trip ID: {trip_id}"
        ).add_to(m)
    
    # Sesuaikan bounding box agar mencakup seluruh titik perjalanan
    if all_lat and all_lon:
        min_lat, max_lat = min(all_lat), max(all_lat)
        min_lon, max_lon = min(all_lon), max(all_lon)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
    out_path = os.path.join(MAP_FOLDER, "all_trips_route.html")
    m.save(out_path)
    print(f"\nSelesai! Peta rute visualisasi berhasil disimpan di:\n{out_path}")

if __name__ == "__main__":
    main()
