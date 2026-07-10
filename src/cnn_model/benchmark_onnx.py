import os
import sys
import time
import numpy as np
import onnxruntime as ort

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_MODEL_PATH = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d", "cnn_1d_model.onnx")

def benchmark_onnx(model_path, num_runs=1000):
    print(f"Loading ONNX Model: {model_path}")
    
    # Initialize ONNX session (CPU execution provider to simulate mobile CPU)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1 # Simulate single-threaded constraint on mobile
    session_options.inter_op_num_threads = 1
    
    session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model Input: {session.get_inputs()[0].name}, Shape: {session.get_inputs()[0].shape}")
    print(f"Model Output: {session.get_outputs()[0].name}, Shape: {session.get_outputs()[0].shape}")
    
    # Dummy input representing raw sensor data [1, 7, 230]
    dummy_input = np.random.randn(1, 7, 230).astype(np.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        session.run([output_name], {input_name: dummy_input})
        
    print(f"Running benchmark for {num_runs} iterations...")
    latencies = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to milliseconds
        
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    model_size_kb = os.path.getsize(model_path) / 1024
    
    print("\n" + "="*40)
    print("ONNX BENCHMARK RESULTS (CPU, 1 Thread)")
    print("="*40)
    print(f"Model Size     : {model_size_kb:.2f} KB")
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"P50 Latency    : {p50_latency:.3f} ms")
    print(f"P95 Latency    : {p95_latency:.3f} ms")
    print(f"P99 Latency    : {p99_latency:.3f} ms")
    print("="*40)
    
    # Output single inference result
    result = session.run([output_name], {input_name: dummy_input})[0]
    print(f"\nSample Prediction Output (Softmax embedded):\n{result}")

if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ERROR: Model not found at {ONNX_MODEL_PATH}")
    else:
        benchmark_onnx(ONNX_MODEL_PATH)
