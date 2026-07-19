import os
import sys
import torch
import numpy as np

# Fix Windows Emoji crash in PyTorch ONNX exporter
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from model import InceptionTime1D

import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
from config import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "models", "cnn_1d")

class MobileInferenceWrapper(torch.nn.Module):
    def __init__(self, base_model, means, stds):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
        
    def forward(self, x):
        # x is [batch_size, channels, seq_len]
        # 1. Global Z-score scaling using embedded constants
        x_scaled = (x - self.means) / self.stds
        
        # 2. Base Model Forward
        logits = self.base_model(x_scaled)
        
        # 3. Softmax Probabilities (Matches MultiClassFocalLoss used in training)
        return torch.softmax(logits, dim=1)

def main():
    import json
    pth_path = os.path.join(MODEL_DIR, "cnn_1d_model.pth")
    classes_path = os.path.join(MODEL_DIR, "cnn_1d_classes.npy")
    
    if not os.path.exists(pth_path):
        logger.error(f"Model weights not found at {pth_path}")
        return
        
    classes = np.load(classes_path)
    
    # Initialize base model with exactly 7 channels
    base_model = InceptionTime1D(in_channels=7, num_classes=len(classes))
    base_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    base_model.eval()
    
    scaler_path = os.path.join(MODEL_DIR, "cnn_1d_scaler_params.json")
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
        
    means = torch.tensor(scaler_params['means'], dtype=torch.float32).view(1, 7, 1)
    stds = torch.tensor(scaler_params['stds'], dtype=torch.float32).view(1, 7, 1)
    
    # Wrap with MobileInferenceWrapper
    model = MobileInferenceWrapper(base_model, means, stds)
    model.eval()
    
    # Dummy input (Batch_size=1, Channels=7, Length=200)
    dummy_input = torch.randn(1, 7, 200, requires_grad=False)
    
    onnx_path = os.path.join(MODEL_DIR, "cnn_1d_model.onnx")
    
    # Export the model
    torch.onnx.export(model,               # model being run
                  dummy_input,             # model input
                  onnx_path,               # where to save the model
                  export_params=True,      # store the trained parameter weights inside the model file
                  opset_version=15,        # stable opset for mobile
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamo=False,              
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})
                                
    logger.info(f"Model berhasil diekspor ke format ONNX di: {onnx_path}")
    
    # Generate thresholds JSON
    p_idx = list(classes).index("Pothole") if "Pothole" in classes else -1
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in classes else -1
    ne_idx = list(classes).index("Non-Event") if "Non-Event" in classes else -1
    
    threshold_config = {
        "pothole_threshold": 0.5,
        "speed_bump_threshold": 0.5,
        "pothole_class_index": int(p_idx),
        "speed_bump_class_index": int(sb_idx),
        "non_event_class_index": int(ne_idx),
        "class_names": list(classes),
        "calibration_method": "softmax",
        "note": "Probabilities are natively outputted by the ONNX model (embedded softmax)."
    }
    thresh_json_path = os.path.join(MODEL_DIR, "cnn_1d_thresholds.json")
    with open(thresh_json_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Thresholds config saved to: {thresh_json_path}")
    
if __name__ == "__main__":
    main()
