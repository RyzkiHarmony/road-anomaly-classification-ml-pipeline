import os
import time
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_ONNX = os.path.join(BASE_DIR, "evaluation", "models", "xgboost", "xgboost_model.onnx")
CNN_ONNX = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d", "cnn_1d_model.onnx")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def benchmark_onnx_model(model_path, dummy_input_shape, num_iterations=1000):
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} tidak ditemukan.")
        return None
        
    # Get model size on disk (KB)
    file_size_kb = os.path.getsize(model_path) / 1024.0
    
    # Initialize ONNX session
    # Force single thread for CPU benchmarking to simulate mobile environment
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Prepare dummy input
    dummy_input = np.random.randn(*dummy_input_shape).astype(np.float32)
    
    # Warmup runs
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_input})
        
    # Benchmark runs
    times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # Convert to ms
        
    mean_time = np.mean(times)
    std_time = np.std(times)
    p95_time = np.percentile(times, 95)
    throughput = 1000.0 / mean_time
    
    return {
        "file_size_kb": file_size_kb,
        "input_shape": input_shape,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "p95_time_ms": p95_time,
        "throughput_fps": throughput
    }

def main():
    print("============================================================")
    print("             RESOURCE & LATENCY BENCHMARK                   ")
    print("============================================================")
    
    # XGBoost expectations: 56 features
    xgb_results = benchmark_onnx_model(XGB_ONNX, (1, 56))
    
    # CNN expectations: 14 channels, 200 samples length
    cnn_results = benchmark_onnx_model(CNN_ONNX, (1, 14, 200))
    
    report_lines = []
    report_lines.append("======================================================================")
    report_lines.append("             ONNX LATENCY & RESOURCE BENCHMARK REPORT                 ")
    report_lines.append("======================================================================")
    report_lines.append("Device: CPU (Single-Threaded Simulation for Mobile Edge)")
    report_lines.append("----------------------------------------------------------------------")
    report_lines.append(f"{'Metrik':<30} | {'XGBoost ONNX':<18} | {'1D-CNN ONNX':<18}")
    report_lines.append("-" * 72)
    
    if xgb_results and cnn_results:
        report_lines.append(f"{'Model Size (KB)':<30} | {xgb_results['file_size_kb']:<18.2f} | {cnn_results['file_size_kb']:<18.2f}")
        report_lines.append(f"{'Input Shape':<30} | {str(xgb_results['input_shape']):<18} | {str(cnn_results['input_shape']):<18}")
        report_lines.append(f"{'Average Latency (ms)':<30} | {xgb_results['mean_time_ms']:<18.4f} | {cnn_results['mean_time_ms']:<18.4f}")
        report_lines.append(f"{'Standard Deviation (ms)':<30} | {xgb_results['std_time_ms']:<18.4f} | {cnn_results['std_time_ms']:<18.4f}")
        report_lines.append(f"{'95th Percentile (ms)':<30} | {xgb_results['p95_time_ms']:<18.4f} | {cnn_results['p95_time_ms']:<18.4f}")
        report_lines.append(f"{'Throughput (preds/sec)':<30} | {xgb_results['throughput_fps']:<18.2f} | {cnn_results['throughput_fps']:<18.2f}")
        
        report_lines.append("======================================================================")
        report_lines.append("\nANALISIS & REKOMENDASI DEPLOYMENT:")
        
        ratio_size = cnn_results['file_size_kb'] / xgb_results['file_size_kb']
        ratio_time = cnn_results['mean_time_ms'] / xgb_results['mean_time_ms']
        
        report_lines.append(f"- **Ukuran Penyimpanan:** Model 1D-CNN ({cnn_results['file_size_kb']:.1f} KB) adalah {ratio_size:.1f}x lebih besar dari XGBoost ({xgb_results['file_size_kb']:.1f} KB).")
        report_lines.append("  Namun, kedua model tergolong sangat kecil (<1 MB) dan sangat aman untuk penyimpanan internal Android.")
        
        report_lines.append(f"- **Kecepatan Inferensi:** Waktu eksekusi rata-rata 1D-CNN ({cnn_results['mean_time_ms']:.2f} ms) adalah {ratio_time:.1f}x dari XGBoost ({xgb_results['mean_time_ms']:.2f} ms).")
        if cnn_results['mean_time_ms'] < 10.0:
            report_lines.append(f"  Meskipun 1D-CNN lebih lambat, waktu inferensinya ({cnn_results['mean_time_ms']:.2f} ms) berada jauh di bawah ambang batas real-time (10 ms).")
            report_lines.append("  Model 1D-CNN mampu memproses hingga 100+ prediksi per detik pada core CPU tunggal, menjadikannya sangat layak dideploy.")
        else:
            report_lines.append("  Model 1D-CNN memerlukan optimisasi thread atau kuantisasi jika ingin menjamin latency < 10 ms pada perangkat low-end.")
            
    else:
        report_lines.append("Error: Gagal memuat atau memproses salah satu model ONNX.")
        
    report_content = "\n".join(report_lines)
    print(report_content)
    
    report_path = os.path.join(REPORT_DIR, "inference_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nLaporan benchmark disimpan di: {report_path}")

if __name__ == "__main__":
    main()
