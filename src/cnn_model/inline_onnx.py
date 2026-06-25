import onnx
import os

model_dir = r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\06_1dcnn\models"
onnx_path = os.path.join(model_dir, "model_1dcnn.onnx")

# Load the model with load_external_data=True so it pulls the weights from .data
model = onnx.load(onnx_path, load_external_data=True)

# Save the model back to the same path, but force internal data
onnx.save_model(model, onnx_path, save_as_external_data=False, all_tensors_to_one_file=True)
print(f"Successfully inlined weights into {onnx_path}!")
