import os
from openvino.tools.pot import IEEngine, DataLoader, Metric, compress_model_weights, create_pipeline
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.algorithms.quantization.accuracy_aware import AccuracyAwareQuantization
import numpy as np
from PIL import Image

# === Paths ===
onnx_dir = "Openvino/Models/onnx_models"
openvino_dir = "Openvino/Models/openvino_models"
models = ["resnet50", "mobilenetv2", "efficientnetb0"]
calib_dataset_dir = "MY_data/train"  # images for calibration
save_dir = "Openvino/Models/openvino_models/int8"

os.makedirs(save_dir, exist_ok=True)

# === Custom DataLoader for POT ===
class ImageFolderDataLoader(DataLoader):
    def __init__(self, img_dir, input_shape=(3, 224, 224)):
        self.img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.img_paths.append(os.path.join(root, f))
        self.input_shape = input_shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB").resize((self.input_shape[1], self.input_shape[2]))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        return { "input": np.expand_dims(img_np, axis=0) }

# === Loop over models ===
for model_name in models:
    print(f"Quantizing {model_name} to INT8...")

    fp32_model_path = os.path.join(openvino_dir, model_name, "fp32", f"{model_name}.xml")
    int8_out_dir = os.path.join(save_dir, model_name)
    os.makedirs(int8_out_dir, exist_ok=True)

    # Load model
    model_config = {"model_name": model_name, "model": fp32_model_path}
    model = load_model(model_config)

    # DataLoader
    data_loader = ImageFolderDataLoader(calib_dataset_dir)

    # Engine
    engine = IEEngine(config={"device": "CPU"}, data_loader=data_loader, metric=None)

    # Algorithm
    algorithms = [
        {
            "name": "DefaultQuantization", 
            "params": {
                "target_device": "CPU", 
                "preset": "performance", 
                "stat_subset_size": 100
            }
        }
    ]

    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model)

    # Save INT8 model
    compress_model_weights(compressed_model, int8_out_dir)
    print(f"{model_name} INT8 model saved at {int8_out_dir}")

print("All models converted to OpenVINO INT8 IR successfully!")
