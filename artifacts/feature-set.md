# Road Anomaly Classification Feature Set
## (Pothole vs Speed Bump - Smartphone on Motor Holder)

---

## 1. Overview

Dokumen ini mendefinisikan **feature set final** untuk membedakan:
- Pothole (impulsive impact)
- Speed bump (structured traversal)

Kondisi sistem:
- Smartphone dipasang pada **holder motor (rigid mount)**
- Data mengandung **high-frequency vibration (engine + road noise)**
- Sampling rate **tidak statis (estimated per trip)**

Fokus utama:
> Menangkap **struktur temporal** dan **distribusi energi sinyal**, bukan hanya magnitude.

---

## 2. Preprocessing Requirements

### 2.1 Signal Alignment
- Gunakan **gravity-aligned vertical acceleration (a_vert)**

### 2.2 Windowing
- Ambil window: **[-1s, +1s] dari peak utama**

### 2.3 Sampling Rate Handling
- Estimasi:

fs = 1 / median(diff(timestamp))

- Semua parameter berbasis waktu (detik), lalu dikonversi ke sampel

### 2.4 Smoothing
- Gaussian smoothing ringan:
- σ ≈ 20–40 ms (bukan dalam sampel tetap)

---

## 3. Core Features (Tier 1 - Wajib)

### 3.1 Event Duration
**Definisi:**
Durasi ketika:

|a_vert| > median(pre-event) + 3 × MAD(pre-event)


**Window baseline:**
- [-1s, -0.3s]

**Insight:**
- Pothole → pendek
- Speed bump → panjang

---

### 3.2 Significant Peak Count
**Kriteria peak:**
- Prominence berbasis MAD
- Minimum separation: 80–150 ms

**Insight:**
- Speed bump → ≥ 2 peak
- Pothole → biasanya 1

---

### 3.3 Inter-Peak Interval
**Definisi:**
Jarak waktu antara dua peak terbesar

**Insight:**
- Speed bump → konsisten (terkait wheelbase/speed)
- Pothole → tidak stabil / tidak ada pasangan

---

### 3.4 Pre/Post Energy Ratio
**Rumus:**

Energy = sum(a_vert^2)

ratio = Energy_pre / Energy_post


**Window:**
- Pre: [-0.5s, 0]
- Post: [0, +0.5s]

**Insight:**
- Speed bump → ~1 (simetris)
- Pothole → tidak simetris

---

### 3.5 FFT High/Low Frequency Ratio
**Rumus:**

ratio = Energy(>15 Hz) / Energy(<5 Hz)


**Insight:**
- Pothole → high frequency dominant
- Speed bump → low frequency dominant

---

### 3.6 Max Jerk
**Rumus:**

jerk = d(a_vert)/dt
max_jerk = max(|jerk|)


**Insight:**
- Pothole → sangat tinggi
- Speed bump → lebih smooth

---

## 4. Shape & Distribution Features (Tier 2)

### 4.1 Kurtosis
- Mengukur “spikiness”
- Pothole → tinggi

---

### 4.2 Skewness
- Mengukur asimetri distribusi
- Pothole → cenderung asimetris

---

### 4.3 Peak-to-Peak Amplitude

max(a_vert) - min(a_vert)

- Hanya fitur pendukung

---

### 4.4 Zero Crossing Rate (ZCR)
- Jumlah crossing terhadap nol

**Insight:**
- Pothole → tinggi (chaotic)
- Speed bump → lebih stabil

---

## 5. Rotational Features (Gyroscope - Tier 3)

### 5.1 Pitch Pattern Consistency
Gunakan axis gyro yang sesuai pitch

**Fitur:**
- jumlah peak gyro
- pola forward → backward

---

### 5.2 Gyro Duration
- Durasi aktivitas gyro signifikan

---

## 6. Normalization Layer (WAJIB)

### 6.1 Trip-level Normalization
- Gunakan median & MAD per trip
- Normalize amplitude-based features

---

### 6.2 Speed Proxy Normalization
Jika tidak ada GPS:
- gunakan envelope low-frequency accel

Digunakan untuk:
- normalisasi duration
- normalisasi inter-peak interval

---

## 7. Final Feature Vector

### Minimal (Baseline Model)
- duration
- peak_count
- inter_peak_interval
- pre_post_energy_ratio
- fft_high_low_ratio
- max_jerk
- kurtosis
- ZCR

---

### Extended (Improved Model)
Tambahkan:
- skewness
- gyro_peak_count
- gyro_duration

---

## 8. Features to Avoid (Low Value)

Jangan mengandalkan:
- mean
- standard deviation
- variance

Alasan:
> Tidak merepresentasikan struktur temporal event

---

## 9. Validation Strategy

Plot distribusi fitur:

- duration vs kurtosis
- inter_peak_interval vs peak_count
- fft_ratio vs max_jerk

Jika kelas tidak terpisah:
> Problem ada di feature design atau data, bukan model

---

## 10. Key Insight

- Pothole = **impulsive, high-frequency, asymmetric**
- Speed bump = **structured, multi-phase, low-frequency**

Feature set harus mencerminkan ini secara eksplisit.

---

## 11. Common Failure Modes

- Over-reliance pada threshold (3G, 5G, dst)
- Parameter berbasis sampel, bukan waktu
- Tidak mempertimbangkan speed variation
- Menganggap spike besar = pothole

---

## 12. Conclusion

Pipeline tanpa feature ini hanya:
> Shock detector

Pipeline dengan feature ini:
> Road anomaly classifier