import os
import sys
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")
XGB_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")

def main():
    # Load OOF data
    y_true_encoded = np.load(os.path.join(MODEL_DIR, "cnn_1d_oof_y_true.npy"))
    y_pred_encoded = np.load(os.path.join(MODEL_DIR, "cnn_1d_oof_y_pred.npy"))
    
    # Load label encoder
    le = joblib.load(os.path.join(XGB_DIR, "xgboost_label_encoder.pkl"))
    classes = le.classes_
    
    y_true = le.inverse_transform(y_true_encoded)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    print("=== OUT-OF-FOLD (OOF) EVALUATION ===")
    print("\n1. CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    
    print("\n2. MACRO F1-SCORE")
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1-Score: {macro_f1:.3f}")
    
    print("\n3. CONFUSION MATRIX")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Print formatted confusion matrix
    print(f"{'True \\ Pred':>15} | " + " | ".join([f"{c:>10}" for c in classes]))
    print("-" * 60)
    for i, true_class in enumerate(classes):
        row = f"{true_class:>15} | " + " | ".join([f"{val:>10}" for val in cm[i]])
        print(row)

if __name__ == "__main__":
    main()
