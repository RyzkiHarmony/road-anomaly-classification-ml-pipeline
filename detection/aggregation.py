import pandas as pd
import numpy as np
import folium

class MapAggregator:
    def __init__(self):
        pass

    def create_damage_map(self, predictions_df, output_html="road_damage_map.html"):
        """
        Menerima DataFrame hasil prediksi dan menggambarkannya dalam bentuk
        SEGMEN jalur (garis rute berwarna berdasarkan kondisi jalan), bukan sekadar Pin.
        
        :param predictions_df: DataFrame hasil tebakan model.
        :param output_html: Nama file output.
        """
        # Hapus baris yang tidak memiliki koordinat valid atau sorting by time
        df_valid = predictions_df.dropna(subset=['lat_mean', 'lon_mean']).copy()
        
        # Urutkan berdasarkan waktu jika kolom window_start ada, agar garis jalurnya tidak zig-zag
        if 'window_start' in df_valid.columns:
            df_valid = df_valid.sort_values(by='window_start')

        if len(df_valid) == 0:
            print("Tidak ada data GPS yang valid untuk dipetakan.")
            return

        # Titik tengah peta
        center_lat = df_valid['lat_mean'].mean()
        center_lon = df_valid['lon_mean'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")

        print("Merakit segmen-segmen jalan berdasarkan kondisi (Model Prediction)...")

        # Logika Penggabungan Titik Menjadi Segmen
        segments = []
        current_segment = []
        current_label = None

        for idx, row in df_valid.iterrows():
            label = row['prediction']
            coord = (row['lat_mean'], row['lon_mean'])
            
            if current_label is None:
                current_label = label
                current_segment.append(coord)
            elif label == current_label:
                # Masih di kondisi/label yang sama, perpanjang jalurnya
                current_segment.append(coord)
            else:
                # Jika status jalan berubah (misal dari Normal -> Rusak)
                # Tambahkan titik baru ini ke koordinat lama agar garisnya menyambung tanpa patah
                current_segment.append(coord)
                segments.append((current_label, current_segment))
                # Mulai segmen baru untuk kondisi baru
                current_segment = [coord]
                current_label = label
                
        # Akhiri rangkaian
        if len(current_segment) > 1:
            segments.append((current_label, current_segment))

        # Menggambar Segmen Rute ke Peta
        for label, path in segments:
            if label == 'Severe Anomaly':
                # Jalan Rusak - Segmen Merah Tebal
                color = 'red'
                weight = 6
                opacity = 0.95
            elif label == 'Normal':
                # Jalan Baik - Segmen Biru / Hijau Tipis
                color = '#3498db' # Light Blue
                weight = 3.5
                opacity = 0.6
            else:
                # Anomali Ringan - Orange
                color = 'orange'
                weight = 5
                opacity = 0.8
                
            # Taruh satu segmen garis di peta
            folium.PolyLine(
                path,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"Kondisi Ruas Ini: {label}"
            ).add_to(m)

        # Anda tetap bisa menaruh Marker Khusus JIKA guncangannya sangat fantastis ekstrim
        # sebagai pelengkap (opsional). Pada kodingan ini difokuskan pada Segmen.
        
        # Simpan Peta
        m.save(output_html)
        print(f"Peta Rute Segmen Jalan telah ditenun & disimpan di: {output_html}")
        return m

if __name__ == "__main__":
    print("Modul MapAggregator (Segment-Based Folium) siap.")
