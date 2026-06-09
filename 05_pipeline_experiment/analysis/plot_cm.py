import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Add parent directory to sys.path to import config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import OUT_FOLDER, BEST_FEATURES

# Paths (relative to parent dir)
model_path = os.path.join(parent_dir, "models", "best_model.pkl")
le_path = os.path.join(parent_dir, "models", "label_encoder.pkl")
data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")

def main():
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("Model or data not found.")
        return

    # Load artifacts
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    df = pd.read_csv(data_path).dropna(subset=['label'])

    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y_true = df['label'].values
    y_true_enc = le.transform(y_true)

    # Predict
    y_pred_enc = model.predict(X)
    classes = le.classes_

    # Confusion Matrix
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Evaluated on All Train Data)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    save_path = os.path.join(OUT_FOLDER, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    
    # Text output
    print("\nText Confusion Matrix:")
    df_cm = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
    print(df_cm.to_string())

if __name__ == "__main__":
    main()
