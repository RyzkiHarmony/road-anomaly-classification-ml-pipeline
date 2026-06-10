import os
import torch
import numpy as np
from sklearn.metrics import classification_report
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '06_1dcnn'))
from model import Lightweight1DCNN

def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '06_1dcnn', "data")
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '06_1dcnn', "models")
    
    # Load Data
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    
    # Load Final Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lightweight1DCNN(in_channels=3, num_classes=len(classes),
                             conv1_filters=32, conv2_filters=64, dropout_rate=0.169).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_1dcnn.pth"), map_location=device))
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Evaluate
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    print("=== Performa pada FULL TRAINING DATA (Memorization Check) ===")
    print(classification_report(y, all_preds, target_names=classes))

if __name__ == "__main__":
    main()
