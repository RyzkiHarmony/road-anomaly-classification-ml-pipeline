import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from config import OUT_FOLDER, get_logger

logger = get_logger(__name__)

def main():
    # 1. Load Artifacts
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    le_path = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    
    if not all(os.path.exists(p) for p in [model_path, le_path, data_path]):
        logger.error("Artifacts (model/data) tidak lengkap. Pastikan train_model.py sudah dijalankan.")
        return

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    df = pd.read_csv(data_path)

    # 2. Prepare Features (Gunakan 10 fitur terbaik yang sama dengan training)
    BEST_FEATURES = [
        "event_duration", "speed_normalized_p2p", "peak_interval_std", 
        "vert_jrk", "kurtosis", "peak_mag", "peak_interval_mean", 
        "skewness", "gyro_roll_energy", "num_peaks_accel"
    ]
    
    # Filter hanya data yang punya fitur lengkap
    df_eval = df.dropna(subset=BEST_FEATURES + ["label"]).copy()
    X = df_eval[BEST_FEATURES].values
    y_true = df_eval["label"].values
    
    # 3. Predict Probabilities
    # Cari indeks untuk kelas 'Pothole'
    classes = list(le.classes_)
    p_idx = classes.index("Pothole")
    
    y_proba = model.predict_proba(X)
    pothole_probs = y_proba[:, p_idx]
    
    # Gunakan threshold optimal dari hasil training sebelumnya (~0.16)
    THRESHOLD = 0.16
    
    # Prediksi berdasarkan threshold khusus Pothole
    # (Jika prob > THRESHOLD, paksa jadi Pothole untuk audit sensitivitas)
    y_pred_custom = []
    for i, prob in enumerate(pothole_probs):
        if prob >= THRESHOLD:
            y_pred_custom.append("Pothole")
        else:
            # Jika di bawah threshold, ambil kelas dengan probabilitas tertinggi lainnya
            # (Tapi kita fokus audit Pothole)
            y_pred_custom.append(le.inverse_transform([np.argmax(y_proba[i])])[0])
            
    df_eval["pred_label"] = y_pred_custom
    df_eval["pothole_confidence"] = pothole_probs

    # 4. Filter Errors
    # False Positives: Model bilang Pothole, Padahal Non-Event
    fp_mask = (df_eval["pred_label"] == "Pothole") & (df_eval["label"] == "Non-Event")
    # False Negatives: Model bilang Bukan Pothole, Padahal Pothole
    fn_mask = (df_eval["pred_label"] != "Pothole") & (df_eval["label"] == "Pothole")
    
    errors_fp = df_eval[fp_mask].copy()
    errors_fn = df_eval[fn_mask].copy()
    
    # 5. Export for Audit
    output_audit_path = os.path.join(OUT_FOLDER, "error_audit_results.csv")
    
    # Gabungkan FP dan FN
    errors_fp["error_type"] = "FALSE_POSITIVE (Model Over-sensitive)"
    errors_fn["error_type"] = "FALSE_NEGATIVE (Model Missed It)"
    
    audit_df = pd.concat([errors_fp, errors_fn])
    
    # Sort berdasarkan confidence agar yang paling "yakin tapi salah" muncul di atas
    audit_df = audit_df.sort_values(by="pothole_confidence", ascending=False)
    
    cols_to_save = [
        "event_id", "trip_id", "time_s", "label", "pred_label", 
        "error_type", "pothole_confidence", "speed_mean", "vertical_energy"
    ]
    audit_df[cols_to_save].to_csv(output_audit_path, index=False)
    
    print("\n" + "="*60)
    print(f"{'ERROR AUDIT SUMMARY':^60}")
    print("="*60)
    print(f"Total False Positives (FP) : {len(errors_fp)}")
    print(f"Total False Negatives (FN) : {len(errors_fn)}")
    print(f"Audit file disimpan di     : {output_audit_path}")
    print("="*60)
    print("\nREKOMENDASI:")
    print("1. Buka file error_audit_results.csv")
    print("2. Cari event_id yang punya pothole_confidence paling tinggi di kelompok FP.")
    print("3. Cek di GUI Labeling: Apakah itu getaran mesin? Atau lubang yang Anda lewatkan?")

if __name__ == "__main__":
    main()
