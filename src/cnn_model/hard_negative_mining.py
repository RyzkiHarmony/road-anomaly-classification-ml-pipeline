import os
import torch
import numpy as np
import joblib

from model import InceptionTime1D

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "cnn_1d")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d")
XGB_MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost")

CONF_THRESHOLD = 0.50

def scale_instance_level(X, eps=1e-8):
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    return (X - mean) / (std + eps)

def main():
    pth_path = os.path.join(MODEL_DIR, "cnn_1d_model.pth")
    
    if not os.path.exists(pth_path):
        logger.error("Model tidak ditemukan. Harus ditraining dulu.")
        return
        
    logger.info("Loading CNN Data...")
    X_cnn_all = np.load(os.path.join(DATA_DIR, "cnn_1d_X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "cnn_1d_y.npy"))
    groups = np.load(os.path.join(DATA_DIR, "cnn_1d_groups.npy"))
    
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    classes = le.classes_
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionTime1D(in_channels=X_cnn_all.shape[1], num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    X_scaled = scale_instance_level(X_cnn_all)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    logger.info("Running Inference to find Hard Negatives...")
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(X_tensor), batch_size=1024, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(all_probs, axis=1)
    pred_labels = le.inverse_transform(preds)
    
    # Kriteria Hard Negative: Ground Truth adalah Non-Event, tetapi diprediksi sebagai Pothole atau Speed Bump
    # dengan confidence di atas CONF_THRESHOLD
    confidence = np.max(all_probs, axis=1)
    
    mask_hn = (y_raw == 'Non-Event') & np.isin(pred_labels, ['Pothole', 'Speed Bump']) & (confidence > CONF_THRESHOLD)
    
    hn_X = X_cnn_all[mask_hn]
    hn_groups = groups[mask_hn]
    hn_y = np.array(["Non-Event"] * len(hn_X))
    
    if len(hn_X) > 0:
        logger.info(f"Ditemukan {len(hn_X)} Hard Negatives (False Positives)!")
        
        hn_X_path = os.path.join(DATA_DIR, "X_hard_negatives.npy")
        hn_y_path = os.path.join(DATA_DIR, "y_hard_negatives.npy")
        hn_g_path = os.path.join(DATA_DIR, "groups_hard_negatives.npy")
        
        np.save(hn_X_path, hn_X)
        np.save(hn_y_path, hn_y)
        np.save(hn_g_path, hn_groups)
        
        logger.info(f"Hard negatives berhasil disimpan di karantina: {hn_X_path}")
        logger.warning("PENTING: Tinjau sampel hard negatives di atas secara manual sebelum digabungkan ke dataset utama untuk mencegah Label Contamination!")
    else:
        logger.info("Tidak ada Hard Negative baru yang ditemukan. Model sudah sempurna atau threshold terlalu tinggi.")

if __name__ == "__main__":
    main()
