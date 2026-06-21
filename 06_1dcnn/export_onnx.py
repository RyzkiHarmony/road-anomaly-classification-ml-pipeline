import os
import sys
import torch
import numpy as np

# Fix Windows Emoji crash in PyTorch ONNX exporter
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from model import Lightweight1DCNN

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_pipeline_experiment'))
from config import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def main():
    pth_path = os.path.join(MODEL_DIR, "best_1dcnn.pth")
    classes_path = os.path.join(MODEL_DIR, "classes.npy")
    
    if not os.path.exists(pth_path):
        logger.error(f"Model weights not found at {pth_path}")
        return
        
    classes = np.load(classes_path)
    
    # Initialize model
    model = Lightweight1DCNN(in_channels=10, num_classes=len(classes),
                             conv1_filters=32, conv2_filters=64, dropout_rate=0.169)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    # Dummy input (Batch_size=1, Channels=10, Length=200)
    dummy_input = torch.randn(1, 10, 200, requires_grad=True)
    
    onnx_path = os.path.join(MODEL_DIR, "model_1dcnn.onnx")
    
    # Export the model
    torch.onnx.export(model,               # model being run
                  dummy_input,             # model input (or a tuple for multiple inputs)
                  onnx_path,               # where to save the model (can be a file or file-like object)
                  export_params=True,      # store the trained parameter weights inside the model file
                  opset_version=18,        # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
                                
    logger.info(f"Model berhasil diekspor ke format ONNX di: {onnx_path}")
    
    # Force inline external data if PyTorch created it
    import onnx
    import glob
    data_files = glob.glob(onnx_path + "*.data")
    if data_files:
        logger.info(f"Menggabungkan external data ({data_files[0]}) ke dalam {onnx_path}...")
        try:
            onnx_model = onnx.load(onnx_path, load_external_data=True)
            onnx.save(onnx_model, onnx_path)
            os.remove(data_files[0])
            logger.info("External data berhasil digabungkan dan file .data dihapus.")
        except Exception as e:
            logger.error(f"Gagal menggabungkan external data: {e}")
if __name__ == "__main__":
    main()
