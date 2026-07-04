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

def main():
    pth_path = os.path.join(MODEL_DIR, "cnn_1d_model.pth")
    classes_path = os.path.join(MODEL_DIR, "cnn_1d_classes.npy")
    
    if not os.path.exists(pth_path):
        logger.error(f"Model weights not found at {pth_path}")
        return
        
    classes = np.load(classes_path)
    
    # Initialize model
    model = InceptionTime1D(in_channels=18, num_classes=len(classes),
                            num_blocks=2, channels=64, bottleneck_channels=16, dropout_rate=0.2)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    # Dummy input (Batch_size=1, Channels=18, Length=200)
    dummy_input = torch.randn(1, 18, 200, requires_grad=True)
    
    onnx_path = os.path.join(MODEL_DIR, "cnn_1d_model.onnx")
    
    # Export the model
    torch.onnx.export(model,               # model being run
                  dummy_input,             # model input (or a tuple for multiple inputs)
                  onnx_path,               # where to save the model (can be a file or file-like object)
                  export_params=True,      # store the trained parameter weights inside the model file
                  opset_version=18,        # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamo=False,              # disable Dynamo exporter to use JIT tracing for dynamic_axes
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
