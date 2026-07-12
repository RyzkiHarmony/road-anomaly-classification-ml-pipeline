# Panduan Integrasi Android: Preprocessing Alignment & ONNX Inference

Dokumen ini menjelaskan spesifikasi input model, penyelarasan (*alignment*) preprocessing, dan menyediakan panduan integrasi untuk menjalankan inferensi model **1D-CNN** secara *real-time* di aplikasi Android menggunakan library **ONNX Runtime Mobile**.

---

## 1. Spesifikasi Input Model ONNX (1D-CNN)

*   **Nama Input:** `input`
*   **Tipe Data:** `Float32`
*   **Dimensi Input:** `[1, 7, 200]` (Batch size = 1, Channels = 7, Sequence length = 200 / 2 detik pada 100Hz)
*   **Nama Output:** `output`
*   **Dimensi Output:** `[1, 3]` (Kelas: `0 = Non-Event`, `1 = Pothole`, `2 = Speed Bump`)

### Urutan 7 Channel Sinyal (Kritis: Harus Sesuai!)

Model hanya membutuhkan 7 sinyal fitur mentah dasar yang dikumpulkan pada **100Hz**:

| Index Channel | Nama Sinyal | Keterangan Preprocessing Dasar |
|:---:|---|---|
| **0** | `speed` | Kecepatan kendaraan dalam m/s (dari GPS) |
| **1** | `a_x` | Akselerasi sumbu X (termasuk gravitasi jika menggunakan Sensor.TYPE_ACCELEROMETER) |
| **2** | `a_y` | Akselerasi sumbu Y |
| **3** | `a_z` | Akselerasi sumbu Z |
| **4** | `gx` | Kecepatan sudut giroskop roll (X-axis) |
| **5** | `gy` | Kecepatan sudut giroskop pitch (Y-axis) |
| **6** | `gz` | Kecepatan sudut giroskop yaw (Z-axis) |

---

## 2. Penyelarasan Preprocessing (Python vs Kotlin)

Penyelarasan pada model versi terbaru telah disederhanakan drastis berkat penggunaan `MobileInferenceWrapper` selama proses ekspor ONNX di Python.

### A. Standardisasi Sinyal (Z-Score Scaling) - Otomatis
Anda **TIDAK PERLU** melakukan Z-Score Scaling secara manual di Kotlin. Parameter *mean* dan *standard deviation* (sebelumnya ada di `cnn_1d_scaler_params.json`) telah ditanam (*embedded*) ke dalam graf arsitektur ONNX. 
Kirimkan saja *raw data sensor* (dalam tipe Float32) langsung ke ONNX.

### B. Aktivasi Probabilitas (Sigmoid) - Otomatis
Sama seperti scaler, fungsi aktivasi Sigmoid untuk mendapatkan skor probabilitas independen dari masing-masing kelas telah ditanam ke dalam model ONNX. Keluaran dari model ini **sudah berwujud probabilitas [0.0 - 1.0]**, sehingga tidak perlu kalkulasi *Softmax* atau *Sigmoid* secara manual.

---

## 3. Implementasi Kode Kotlin (Android Studio)

Tambahkan dependency ONNX Runtime ke `app/build.gradle`:
```groovy
dependencies {
    implementation 'ai.onnxruntime:onnxruntime-android:1.16.0' // atau versi stabil terbaru
}
```

Berikut adalah contoh referensi kelas `OnnxModelRunner` yang minimalis namun optimal:

```kotlin
package com.pemalang.roaddamage.domain

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

class OnnxModelRunner(private val context: Context) {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    fun initialize() {
        ortEnvironment = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open("cnn_1d_model.onnx").readBytes()
        
        val options = OrtSession.SessionOptions()
        try {
            options.addNnapi() // Aktifkan akselerasi perangkat keras
        } catch (e: Exception) {
            Log.w("OnnxRunner", "NNAPI not available, fallback to CPU.")
        }
        
        ortSession = ortEnvironment?.createSession(modelBytes, options)
    }

    suspend fun predict(flatData: FloatArray): FloatArray = withContext(Dispatchers.Default) {
        val env = ortEnvironment ?: throw IllegalStateException("ONNX Environment not initialized")
        val session = ortSession ?: throw IllegalStateException("ONNX Session not initialized")

        // Bentuk input tensor: [Batch=1, Channels=7, Length=200]
        val shape = longArrayOf(1, 7, 200)
        
        val byteBuffer = java.nio.ByteBuffer.allocateDirect(flatData.size * 4)
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())
        val floatBuffer = byteBuffer.asFloatBuffer()
        floatBuffer.put(flatData)
        floatBuffer.rewind()
        
        val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)
        
        try {
            val inputName = session.inputNames.iterator().next()
            val inputs = mapOf(inputName to tensor)
            
            val result = session.run(inputs)
            try {
                // Output langsung berupa probabilitas, tak perlu Sigmoid manual
                val outputTensor = result.iterator().next().value as OnnxTensor
                val outFloatBuffer = outputTensor.floatBuffer
                val probs = FloatArray(3) // [0: Non-Event, 1: Pothole, 2: Speed Bump]
                outFloatBuffer.get(probs)
                
                return@withContext probs
            } finally {
                result.close()
            }
        } finally {
            tensor.close()
        }
    }

    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
    }
}
```

---

## 4. Checklist Validasi untuk Android Developer

Sebelum merilis aplikasi, pastikan developer Android memverifikasi hal berikut:

- [ ] **Frekuensi Sensor (100Hz):** Pembacaan sensor Android dikumpulkan atau di-*resample* menjadi `10ms` secara konstan (menggunakan interpolasi untuk menambal *gap* temporal jika diperlukan).
- [ ] **Urutan Axis GIROSKOP & AKSELEROMETER:** Pastikan axis giroskop (`gx`, `gy`, `gz`) dan akselerometer terorientasi secara konsisten, lalu urutannya dimasukkan ke *flat array* sama persis seperti tabel di atas.
- [ ] **Filter Diam/Berhenti:** Lewati proses *inferensi* (jangan panggil `predict()`) apabila `speed` GPS mencatat laju di bawah `1.0 m/s` untuk menghindari *false positive* saat macet / berhenti.
