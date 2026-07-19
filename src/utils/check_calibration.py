import numpy as np
import os

# Sesuaikan dengan path Anda jika berbeda
MODEL_DIR = r"evaluation\models\cnn_1d"
CLASSES = np.load(os.path.join(MODEL_DIR, "cnn_1d_classes.npy"))
pothole_idx = list(CLASSES).index("Pothole")

y_true = np.load(os.path.join(MODEL_DIR, "cnn_1d_oof_y_true.npy"))
y_pred = np.load(os.path.join(MODEL_DIR, "cnn_1d_oof_y_pred.npy"))
y_proba = np.load(os.path.join(MODEL_DIR, "cnn_1d_oof_y_proba.npy"))

# Cari semua sampel di mana:
# 1. Label aslinya adalah Pothole
# 2. Model MENEBAK BENAR (Argmax = Pothole)
correct_pothole_mask = (y_true == pothole_idx) & (y_pred == pothole_idx)
correct_pothole_probas = y_proba[correct_pothole_mask, pothole_idx]

total_correct = len(correct_pothole_probas)
if total_correct == 0:
    print("Model tidak pernah menebak Pothole dengan benar sama sekali.")
else:
    avg_prob = np.mean(correct_pothole_probas)
    under_50 = np.sum(correct_pothole_probas < 0.5)
    under_30 = np.sum(correct_pothole_probas < 0.3)
    
    print(f"Total True Positive Pothole (Argmax benar) : {total_correct} sampel")
    print(f"Rata-rata probabilitas mentah (Sigmoid/Softmax) : {avg_prob:.4f}")
    print(f"Jumlah yang probabilitasnya di bawah 0.50  : {under_50} sampel ({(under_50/total_correct)*100:.1f}%)")
    print(f"Jumlah yang probabilitasnya di bawah 0.30  : {under_30} sampel ({(under_30/total_correct)*100:.1f}%)")
    print("-" * 50)
    
    if avg_prob < 0.5:
        print("KESIMPULAN SENIOR ML ENGINEER: Bencana Kalibrasi! (Poor Calibration)")
        print("Model Anda menebak benar HANYA karena probabilitas kelas lain lebih hancur lagi.")
        print("Di aplikasi Android (yang menggunakan threshold misal > 0.6), pothole ini TIDAK AKAN TERDETEKSI.")
    else:
        print("KESIMPULAN: Probabilitas terkalibrasi dengan baik. Model cukup percaya diri.")
