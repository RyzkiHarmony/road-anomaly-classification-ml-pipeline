import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
from sklearn.preprocessing import LabelEncoder

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

XGB_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "xgboost")
CNN_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "cnn_1d")
XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation", "reports", "cnn_1d")

from model import InceptionTime1D
from data_utils import get_stratified_group_split

def scale_instance_level(X, eps=1e-8):
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    return (X - mean) / (std + eps)

def main():
    print("Loading data...")
    
    print("Loading CNN Data...")
    X_cnn_all = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_X.npy"))
    y_raw = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_y.npy"))
    groups = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_groups.npy"))
    event_ids = np.load(os.path.join(CNN_DATA_DIR, "cnn_1d_event_ids.npy"))
    
    # Load Label Encoder
    le = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_label_encoder.pkl"))
    classes = le.classes_
    
    _, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    test_mask = np.isin(groups, test_groups_list)
    
    # Filter for test set
    X_cnn_test = X_cnn_all[test_mask]
    y_test = y_raw[test_mask]
    event_ids_test = event_ids[test_mask]
    groups_test = groups[test_mask]
    
    df_test = pd.DataFrame({
        'event_id': event_ids_test,
        'trip_id': groups_test,
        'label': y_test
    })
    
    # Preprocess CNN Data
    X_cnn_scaled = scale_instance_level(X_cnn_test)
    X_cnn_tensor = torch.tensor(X_cnn_scaled, dtype=torch.float32)
    
    # Load Model
    print("Loading CNN Model...")
    cnn_model = InceptionTime1D(in_channels=X_cnn_tensor.shape[1], num_classes=len(classes))
    cnn_model.load_state_dict(torch.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_model.pth"), map_location=torch.device('cpu')))
    cnn_model.eval()
    
    # Inference
    print("Running Inference...")
    with torch.no_grad():
        outputs = cnn_model(X_cnn_tensor)
        cnn_probas = torch.softmax(outputs, dim=1).numpy()
        
    cnn_preds = np.argmax(cnn_probas, axis=1)
    cnn_pred_labels = le.inverse_transform(cnn_preds)
    
    df_test['cnn_prediction'] = cnn_pred_labels
    df_test['cnn_confidence'] = np.max(cnn_probas, axis=1)
    
    # Filter False Positives (True Label = Non-Event, Predicted = Pothole or Speed Bump)
    fp_mask = (df_test['label'] == 'Non-Event') & (df_test['cnn_prediction'].isin(['Pothole', 'Speed Bump']))
    df_fp = df_test[fp_mask].copy()
    
    # Select important columns to show
    cols_to_save = ['trip_id', 'time_s', 'lat', 'lon', 'label', 'cnn_prediction', 'cnn_confidence', 'speed']
    # If speed is not a column in df, maybe we can ignore it
    cols_to_save = [c for c in cols_to_save if c in df_fp.columns]
    
    df_fp = df_fp[cols_to_save]
    df_fp = df_fp.sort_values(by=['cnn_confidence'], ascending=False)
    
    output_path = os.path.join(OUTPUT_DIR, "cnn_false_positives.csv")
    df_fp.to_csv(output_path, index=False)
    
    print(f"Extracted {len(df_fp)} False Positive cases.")
    print(f"Saved to: {output_path}")
    print("\nTop 10 Most Confident False Positives:")
    print(df_fp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
