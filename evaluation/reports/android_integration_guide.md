# Panduan Integrasi Android: Preprocessing Alignment & ONNX Inference

Dokumen ini menjelaskan spesifikasi input model, penyelarasan (*alignment*) preprocessing, dan menyediakan kelas bantuan **Kotlin** untuk menjalankan inferensi model **1D-CNN** secara *real-time* di aplikasi Android menggunakan library **ONNX Runtime Mobile**.

---

## 1. Spesifikasi Input Model ONNX (1D-CNN)

*   **Nama Input:** `input`
*   **Tipe Data:** `Float32`
*   **Dimensi Input:** `[1, 14, 200]` (Batch size = 1, Channels = 14, Sequence length = 200 / 2 detik pada 100Hz)
*   **Nama Output:** `output`
*   **Dimensi Output:** `[1, 3]` (Kelas: `0 = Non-Event`, `1 = Pothole`, `2 = Speed Bump`)

### Urutan 14 Channel Sinyal (Kritis: Harus Sesuai!)

| Index Channel | Nama Sinyal | Keterangan Preprocessing |
|:---:|---|---|
| **0** | `a_vertical` | Akselerasi vertikal linear (gravitasi dibuang) |
| **1** | `a_horizontal` | Amplitudo akselerasi horizontal (magnitudo XY) |
| **2** | `speed` | Kecepatan kendaraan dalam m/s (dari GPS) |
| **3** | `a_vertical_crest_factor` | Faktor puncak akselerasi vertikal (rolling window) |
| **4** | `a_vertical_jerk` | Jerk akselerasi vertikal (turunan pertama terhadap waktu) |
| **5** | `gx` | Kecepatan sudut giroskop roll (X-axis) |
| **6** | `gy` | Kecepatan sudut giroskop pitch (Y-axis) |
| **7** | `gz` | Kecepatan sudut giroskop yaw (Z-axis) |
| **8** | `g_roll_accel` | Akselerasi sudut roll (turunan pertama `gx`) |
| **9** | `g_pitch_accel` | Akselerasi sudut pitch (turunan pertama `gy`) |
| **10** | `a_vertical_rms` | Root Mean Square (RMS) akselerasi vertikal (rolling window) |
| **11** | `a_vertical_zcr` | Zero Crossing Rate akselerasi vertikal (rolling window) |
| **12** | `a_horizontal_rms` | Root Mean Square (RMS) akselerasi horizontal (rolling window) |
| **13** | `energy_ratio_vh` | Rasio energi vertikal terhadap horizontal: `vertical_rms / (horizontal_rms + 1e-6)` |

---

## 2. Penyelarasan Preprocessing (Python vs Kotlin)

Untuk mencegah perbedaan performa di ponsel (*training-serving skew*), Anda wajib mengimplementasikan formula penyelarasan berikut di Kotlin:

### A. Standardisasi Sinyal (Z-Score Scaling)
Sebelum dimasukkan ke ONNX, setiap titik data pada channel $c$ harus distandardisasi menggunakan parameter dari `cnn_1d_scaler_params.json`:
$$x_{\text{scaled}} = \frac{x_{\text{raw}} - \mu_c}{\sigma_c}$$

### B. Perhitungan Jerk (Turunan Sinyal)
Gunakan delta waktu ($\Delta t$) nyata antara dua pembacaan sensor (idealnya 10ms atau 0.01 detik):
$$\text{Jerk}_t = \frac{a_t - a_{t-1}}{\Delta t}$$

---

## 3. Implementasi Kode Kotlin (Android Studio)

Tambahkan dependency ONNX Runtime ke `app/build.gradle`:
```groovy
dependencies {
    implementation 'ai.onnxruntime:onnxruntime-android:1.16.0' // atau versi stabil terbaru
}
```

Berikut adalah kelas Helper Kotlin lengkap yang mengurus preprocessing, scaling, dan inferensi ONNX:

```kotlin
package com.skripsi.roaddetection

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.sqrt

class RoadAnomalyDetector(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Parameter standardisasi (Disinkronkan dari cnn_1d_scaler_params.json)
    private val channelMeans = floatArrayOf(
        0.0524f,  // a_vertical
        0.1235f,  // a_horizontal
        7.8598f,  // speed
        3.4418f,  // a_vertical_crest_factor
        196.94f,  // a_vertical_jerk
        0.0012f,  // gx
        -0.0045f, // gy
        0.0008f,  // gz
        1.205f,   // g_roll_accel
        -0.450f,  // g_pitch_accel
        1.528f,   // a_vertical_rms
        0.142f,   // a_vertical_zcr
        0.825f,   // a_horizontal_rms
        13.910f   // energy_ratio_vh
    )
    
    private val channelStds = floatArrayOf(
        1.245f,   // a_vertical
        0.512f,   // a_horizontal
        3.708f,   // speed
        4.149f,   // a_vertical_crest_factor
        122.34f,  // a_vertical_jerk
        0.410f,   // gx
        0.380f,   // gy
        0.290f,   // gz
        15.82f,   // g_roll_accel
        8.24f,    // g_pitch_accel
        0.897f,   // a_vertical_rms
        0.095f,   // a_vertical_zcr
        0.506f,   // a_horizontal_rms
        28.318f   // energy_ratio_vh
    )

    // Threshold Optimal Hasil Tuning & Calibration
    companion object {
        const val THRESHOLD_POTHOLE = 0.5500f
        const val THRESHOLD_SPEED_BUMP = 0.5000f
    }

    init {
        // Load model dari folder assets Android
        val modelBytes = context.assets.open("cnn_1d_model.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    /**
     * Menjalankan klasifikasi pada satu jendela sensor berukuran 2 detik (200 sampel).
     * @param rawData Array 2D berdimensi [14][200] berisi sinyal mentah preprocessing.
     * @return String label hasil deteksi ("Non-Event", "Pothole", atau "Speed Bump").
     */
    fun detectAnomaly(rawData: Array<FloatArray>): String {
        require(rawData.size == 14) { "Data input harus memiliki 14 channels" }
        require(rawData[0].size == 200) { "Panjang jendela harus tepat 200 sampel" }

        // 1. Jalankan Standardisasi (Z-score scaling)
        val scaledData = FloatBuffer.allocate(1 * 14 * 200)
        for (c in 0 until 14) {
            val mean = channelMeans[c]
            val std = channelStds[c]
            for (t in 0 until 200) {
                val scaledVal = (rawData[c][t] - mean) / std
                scaledData.put(scaledVal)
            }
        }
        scaledData.rewind()

        // 2. Wrap ke OnnxTensor
        val inputShape = longArrayOf(1, 14, 200)
        val inputTensor = OnnxTensor.createTensor(env, scaledData, inputShape)

        // 3. Jalankan Inference
        val inputs = mapOf("input" to inputTensor)
        session.execute(inputs).use { results ->
            val outputTensor = results[0] as OnnxTensor
            val logits = (outputTensor.value as Array<FloatArray>)[0]

            // 4. Hitung Softmax untuk probabilitas kelas
            val probs = softmax(logits)

            // 5. Terapkan Threshold Terkalibrasi untuk Klasifikasi
            val probNonEvent = probs[0]
            val probPothole = probs[1]
            val probSpeedBump = probs[2]

            val potholeTriggered = probPothole >= THRESHOLD_POTHOLE
            val speedBumpTriggered = probSpeedBump >= THRESHOLD_SPEED_BUMP

            return when {
                potholeTriggered && speedBumpTriggered -> {
                    if (probPothole >= probSpeedBump) "Pothole" else "Speed Bump"
                }
                potholeTriggered -> "Pothole"
                speedBumpTriggered -> "Speed Bump"
                else -> "Non-Event"
            }
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        
        var sum = 0.0f
        val exp = FloatArray(logits.size)
        for (i in logits.indices) {
            exp[i] = Math.exp((logits[i] - max).toDouble()).toFloat()
            sum += exp[i]
        }
        for (i in exp.indices) {
            exp[i] /= sum
        }
        return exp
    }

    fun close() {
        session.close()
        env.close()
    }
}
```

---

## 4. Checklist Validasi untuk Android Developer

Sebelum merilis aplikasi, pastikan developer Android memverifikasi hal berikut:

- [ ] **Frekuensi Sensor (100Hz):** Pembacaan sensor Android dikumpulkan setiap `10ms` secara konstan (menggunakan buffer ring/interpolasi jika rate hardware berfluktuasi).
- [ ] **Urutan Axis GIROSKOP:** Pastikan axis giroskop (`gx`, `gy`, `gz`) telah terorientasi secara fisik sama dengan orientasi smartphone yang terpasang pada sepeda motor.
- [ ] **Kausalitas Low-Pass Filter:** Gravity removal filter di Android harus bertipe *causal* (tidak boleh menggunakan data di masa depan/non-causal filter) agar tidak memicu delay deteksi.
