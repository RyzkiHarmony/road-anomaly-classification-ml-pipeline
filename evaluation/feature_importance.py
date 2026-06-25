import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")
REPORT_DIR = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

from model import Lightweight1DCNN

def get_stratified_group_split(groups, y_raw, train_ratio=0.7):
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_idx = np.array([class_to_idx[val] for val in y_raw])

    group_names = np.unique(groups)
    group_counts = {g: np.array([np.sum(y_idx[groups == g] == i) for i in range(len(unique_classes))]) for g in group_names}
    total_counts = np.sum(list(group_counts.values()), axis=0)

    train_groups = set()
    test_groups = set()
    current_train = np.zeros(len(unique_classes))

    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)

    return list(train_groups), list(test_groups)

def run_xgb_importance():
    print("\nMenganalisis Feature Importance XGBoost...")
    # Load model
    model = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl"))
    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)
        
    # Get built-in importance (Gain)
    importance_scores = model.feature_importances_
    
    xgb_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_Gain": importance_scores
    }).sort_values(by="Importance_Gain", ascending=False).reset_index(drop=True)
    
    return xgb_imp

def run_cnn_importance():
    print("\nMenganalisis Permutation Feature Importance 1D-CNN...")
    # Load data
    X_path = os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy")
    groups_path = os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy")
    
    X_all = np.load(X_path)
    y_raw_all = np.load(y_path)
    groups_all = np.load(groups_path)
    
    _, test_groups_list = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.7)
    test_mask = np.isin(groups_all, test_groups_list)
    
    X_test_np = X_all[test_mask]
    y_test_raw = y_raw_all[test_mask]
    
    classes = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_classes.npy"), allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    y_test = le.transform(y_test_raw)
    
    # Load scaler parameters
    with open(os.path.join(CNN_MODEL_DIR, "cnn_1d_scaler_params.json"), "r") as f:
        scaler_params = json.load(f)
    means = np.array(scaler_params["means"]).reshape(1, -1, 1)
    stds = np.array(scaler_params["stds"]).reshape(1, -1, 1)
    
    X_test_scaled = (X_test_np - means) / stds
    
    # Load Model
    model = Lightweight1DCNN(in_channels=14, num_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    model.eval()
    
    CHANNELS = [
        "a_vertical", "a_horizontal", "speed", 
        "a_vertical_crest_factor", "a_vertical_jerk",
        "gx", "gy", "gz", 
        "g_roll_accel", "g_pitch_accel",
        "a_vertical_rms", "a_vertical_zcr",
        "a_horizontal_rms", "energy_ratio_vh"
    ]
    
    # Helper to calculate prediction and macro F1
    def get_f1(X_eval):
        X_tensor = torch.tensor(X_eval, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).numpy()
        # Calculate macro F1 of minority classes (Pothole & Speed Bump)
        f1_p = f1_score(y_test, preds, labels=[1], average='macro', zero_division=0)
        f1_sb = f1_score(y_test, preds, labels=[2], average='macro', zero_division=0)
        return (f1_p + f1_sb) / 2.0
    
    baseline_f1 = get_f1(X_test_scaled)
    print(f"  Baseline Macro F1 (Minority Classes): {baseline_f1:.4f}")
    
    # Permute each channel
    rng = np.random.default_rng(42)
    importance_results = []
    
    for idx, channel_name in enumerate(CHANNELS):
        X_permuted = X_test_scaled.copy()
        
        # Shuffle across the batch to destroy context for this channel
        shuffled_indices = rng.permutation(len(X_permuted))
        X_permuted[:, idx, :] = X_permuted[shuffled_indices, idx, :]
        
        permuted_f1 = get_f1(X_permuted)
        f1_drop = baseline_f1 - permuted_f1
        
        importance_results.append({
            "Channel": channel_name,
            "F1_Drop": f1_drop,
            "New_F1": permuted_f1
        })
        
    cnn_imp = pd.DataFrame(importance_results).sort_values(by="F1_Drop", ascending=False).reset_index(drop=True)
    return cnn_imp

def main():
    xgb_imp = run_xgb_importance()
    cnn_imp = run_cnn_importance()
    
    # ─── 1. WRITE REPORT ───
    report_path = os.path.join(REPORT_DIR, "feature_importance_report.txt")
    with open(report_path, "w") as f:
        f.write("============================================================\n")
        f.write("             FEATURE IMPORTANCE REPORT                      \n")
        f.write("============================================================\n\n")
        
        f.write("--- TOP 15 XGBOOST FEATURES (BY GAIN) ---\n")
        f.write(xgb_imp.head(15).to_string(index=True) + "\n\n")
        
        f.write("--- 1D-CNN CHANNEL IMPORTANCE (BY PERMUTATION F1 DROP) ---\n")
        f.write(cnn_imp.to_string(index=True) + "\n")
        
    print(f"\nLaporan feature importance disimpan di: {report_path}")
    
    # ─── 2. PLOT VISUALIZATION ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # XGBoost Plot (Top 15)
    xgb_top = xgb_imp.head(15)
    axes[0].barh(xgb_top["Feature"][::-1], xgb_top["Importance_Gain"][::-1], color='teal')
    axes[0].set_title("XGBoost Top 15 Features (Gain)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Importance Gain")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # CNN Plot
    axes[1].barh(cnn_imp["Channel"][::-1], cnn_imp["F1_Drop"][::-1], color='crimson')
    axes[1].set_title("1D-CNN Channel Importance (Permutation F1 Drop)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("F1-score Drop (Higher = More Important)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafik feature importance disimpan di: {plot_path}")

if __name__ == "__main__":
    main()
