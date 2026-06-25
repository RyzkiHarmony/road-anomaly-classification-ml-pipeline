import onnxruntime as ort
import numpy as np

model_path = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\evaluation\models\cnn_1d\cnn_1d_model.onnx"
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")
print(f"Input shape: {session.get_inputs()[0].shape}")

# Test 1: All zeros
dummy_input = np.zeros((1, 14, 200), dtype=np.float32)
logits = session.run(None, {input_name: dummy_input})[0][0]
print(f"Logits (all zeros): {logits}")
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
print(f"Probs (all zeros): {probs}")

# Test 2: Extreme values resembling CSV (e.g., speed=1.45, aVertical=10.0)
dummy_input = np.zeros((1, 14, 200), dtype=np.float32)
dummy_input[0, 0, :] = 10.0 # aVertical
dummy_input[0, 1, :] = 2.0  # aHorizontal
dummy_input[0, 2, :] = 1.45 # speed
# Tambahkan dummy crest factor dan jerk
dummy_input[0, 3, :] = 5.0  # a_vertical_crest_factor
dummy_input[0, 4, :] = 120.0 # a_vertical_jerk
# Tambahkan dummy gyro
dummy_input[0, 5, :] = 1.2   # gx
dummy_input[0, 6, :] = -0.5  # gy
dummy_input[0, 7, :] = 0.1   # gz
dummy_input[0, 8, :] = 15.0  # g_roll_accel
dummy_input[0, 9, :] = -5.0  # g_pitch_accel
# Sisa channel (10-13): a_vertical_rms, a_vertical_zcr, a_horizontal_rms, energy_ratio_vh
dummy_input[0, 10, :] = 2.0  # a_vertical_rms
dummy_input[0, 11, :] = 0.2  # zcr
dummy_input[0, 12, :] = 0.5  # a_horizontal_rms
dummy_input[0, 13, :] = 4.0  # energy_ratio_vh
logits = session.run(None, {input_name: dummy_input})[0][0]
print(f"\nLogits (rough road, low speed): {logits}")
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
print(f"Probs: {probs}")

def classify_with_thresholds(probs):
    # Urutan kelas: 0 = Non-Event, 1 = Pothole, 2 = Speed Bump
    THRESHOLD_POTHOLE = 0.4507
    THRESHOLD_SPEED_BUMP = 0.7622
    
    if probs[1] >= THRESHOLD_POTHOLE:
        return "Pothole"
    elif probs[2] >= THRESHOLD_SPEED_BUMP:
        return "Speed Bump"
    else:
        return "Non-Event"

print(f"Prediction: {classify_with_thresholds(probs)}")

# Test 3: Rough road, HIGH speed
dummy_input[0, 2, :] = 10.0 # speed = 36 km/h
logits = session.run(None, {input_name: dummy_input})[0][0]
print(f"\nLogits (rough road, HIGH speed): {logits}")
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
print(f"Probs: {probs}")
print(f"Prediction: {classify_with_thresholds(probs)}")

# Test 4: Flat road, high speed
dummy_input[0, 0, :] = 0.0 # aVertical
dummy_input[0, 1, :] = 0.0 # aHorizontal
dummy_input[0, 2, :] = 10.0 # speed
dummy_input[0, 3, :] = 1.0  # crest factor default
dummy_input[0, 4, :] = 0.0  # jerk default
logits = session.run(None, {input_name: dummy_input})[0][0]
print(f"\nLogits (flat road, high speed): {logits}")
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)
print(f"Probs: {probs}")
print(f"Prediction: {classify_with_thresholds(probs)}")

