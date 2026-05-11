# Roadmap: Road Surface Classification (RSC)
**Transitioning from Event-Detection to Continuous State Classification**

## 1. Executive Summary
Sistem saat ini dirancang untuk mendeteksi "titik" (Pothole/Speed Bump). Untuk mendeteksi permukaan jalan (Gravel, Asphalt, Dirt), sistem harus berubah menjadi sistem klasifikasi kontinu berbasis jendela waktu (Sliding Windows). Fitur S-Tier yang sudah diimplementasikan (FFT, Kurtosis, ZCR) adalah fondasi utama untuk transisi ini.

## 2. Perubahan Arsitektur Data
### A. Dari Trigger ke Sliding Window
*   **Current:** Menunggu puncak akselerasi > Threshold -> Ekstrak fitur.
*   **Future:** Membagi seluruh data sensor menjadi jendela tetap (misal: 2 detik) dengan overlap 50% (setiap 1 detik ada prediksi baru).
*   **Benefit:** Seluruh meter perjalanan akan terklasifikasi, bukan hanya saat ada guncangan keras.

### B. Segment-Based Labeling
*   Strategi pelabelan harus berubah dari `timestamp` tunggal menjadi `start_time` dan `end_time`.
*   Usulan Kelas:
    1. `Smooth_Asphalt`: Vibrasi rendah, Kurtosis rendah.
    2. `Rough_Asphalt`: Vibrasi menengah, ZCR stabil.
    3. `Gravel_Broken`: Energi Gyro tinggi, FFT High-Freq dominan.
    4. `Dirt_Road`: Guncangan amplitudo besar tapi frekuensi rendah (tumpul).

## 3. Analisis Relevansi Fitur S-Tier
Fitur yang sudah ada sangat kompatibel untuk RSC:
| Fitur | Relevansi RSC | Target Permukaan |
| :--- | :--- | :--- |
| **FFT High/Low Ratio** | Sangat Tinggi | Membedakan Kerikil (High) vs Aspal (Low). |
| **ZCR (Zero Crossing Rate)** | Tinggi | Mendeteksi tekstur kekasaran permukaan. |
| **Kurtosis** | Tinggi | Membedakan jalan berlubang sporadis vs jalan rusak kontinu. |
| **Gyro Energy** | Sedang | Mendeteksi instabilitas kendaraan pada jalan tanah/bergelombang. |
| **Vertical Energy** | Tinggi | Indikator utama kualitas jalan secara keseluruhan. |

## 4. Modifikasi Kode yang Diperlukan
### A. `feature_extraction.py`
*   Menonaktifkan filter `peak_detection` sebagai pemicu.
*   Menambahkan fungsi `generate_fixed_windows(df, window_size, overlap)` untuk memproses seluruh dataset secara sekuensial.

### B. `manual_labeling_per_trip.py`
*   Memperbarui UI agar pengguna bisa menarik rentang waktu pada grafik untuk memberikan label "Surface".

## 5. Strategi Implementasi (The "Senior" Way)
Jangan membuat model baru dari nol. Gunakan pendekatan **"Two-Stage Detection"**:
1.  **Stage 1 (Surface Classifier):** Menentukan jenis jalan (Gravel/Asphalt).
2.  **Stage 2 (Anomaly Detector):** Jika Stage 1 mendeteksi "Asphalt", aktifkan detektor Pothole. Jika Stage 1 mendeteksi "Gravel", gunakan threshold Pothole yang lebih tinggi agar tidak terkena False Positive.

---
**Prepared by Antigravity (Senior ML Engineer)**
*Targeting Robustness, Scalability, and Scientific Validity.*
