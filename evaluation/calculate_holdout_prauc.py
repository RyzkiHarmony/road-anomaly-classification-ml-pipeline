import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedGroupKFold
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cnn_model.model import InceptionTime1D

def get_stratified_group_split(groups, y, train_ratio=0.8, random_state=42):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
        # The first split will give roughly 80/20
        return np.unique(groups[train_idx]), np.unique(groups[val_idx])

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

DATA_DIR = "data/processed/cnn_1d"
MODEL_DIR = "evaluation/models/cnn_1d"
REPORT_DIR = "evaluation/reports/cnn_1d"
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    X_path = os.path.join(DATA_DIR, "cnn_1d_X.npy")
    y_path = os.path.join(DATA_DIR, "cnn_1d_y.npy")
    groups_path = os.path.join(DATA_DIR, "cnn_1d_groups.npy")
    classes_path = os.path.join(MODEL_DIR, "cnn_1d_classes.npy")
    scaler_path = os.path.join(MODEL_DIR, "cnn_1d_scaler_params.json")
    model_path = os.path.join(MODEL_DIR, "cnn_1d_model.pth")
    
    if not all(os.path.exists(p) for p in [X_path, y_path, groups_path, classes_path, scaler_path, model_path]):
        print("Missing required files.")
        return

    X_all = np.load(X_path)
    y_raw_all = np.load(y_path)
    groups_all = np.load(groups_path)
    classes = np.load(classes_path)
    
    dev_groups, test_groups = get_stratified_group_split(groups_all, y_raw_all, train_ratio=0.8)
    test_mask = np.isin(groups_all, test_groups)
    X_test_np = X_all[test_mask]
    y_test_raw = y_raw_all[test_mask]
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = np.array([class_to_idx[y] for y in y_test_raw])
    
    with open(scaler_path, 'r') as f:
        scaler = json.load(f)
    means = np.array(scaler['means']).reshape(1, X_test_np.shape[1], 1)
    stds = np.array(scaler['stds']).reshape(1, X_test_np.shape[1], 1)
    
    X_test_scaled = (X_test_np - means) / stds
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    best_params_path = "src/cnn_model/best_optuna_params.json"
    with open(best_params_path, 'r') as f:
        bp = json.load(f)
    
    device = torch.device("cpu")
    model = InceptionTime1D(in_channels=X_test_np.shape[1], num_classes=len(classes),
                            num_blocks=3, channels=bp['channels'], 
                            bottleneck_channels=bp['channels']//4, 
                            dropout_rate=bp['dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dataset = SimpleDataset(X_test_tensor, torch.tensor(y_test, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    test_probas = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            test_probas.extend(probs.cpu().numpy())
    test_probas = np.array(test_probas)
    
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pr_aucs = {}
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], test_probas[:, i])
        pr_auc = average_precision_score(y_test_bin[:, i], test_probas[:, i])
        pr_aucs[c] = pr_auc
        
        ax.plot(recall, precision, lw=2, label=f'{c} (PR-AUC = {pr_auc:.3f})')
        print(f"Class: {c} | PR-AUC: {pr_auc:.4f}")
        
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision-Recall Curve (Holdout Test Set)', fontweight='bold', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_path = os.path.join(REPORT_DIR, "cnn_1d_pr_curve_holdout.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"PR Curve saved to {out_path}")

if __name__ == "__main__":
    main()
