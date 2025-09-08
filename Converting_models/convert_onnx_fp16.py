import os
import onnx
from onnxconverter_common import float16

# Paths to your original FP32 ONNX models
original_models = [
    "Onnx_Models/resnet50_fruits.onnx",
    "Onnx_Models/mobilenetv2_fruits.onnx",
    "Onnx_Models/efficientnetb0_fruits.onnx"
]

# Directory to save FP16 quantized models
fp16_dir = "Onnx_Models/fp16"   
os.makedirs(fp16_dir, exist_ok=True)

for model_fp32 in original_models:
    base_name = os.path.basename(model_fp32).replace(".onnx", "")
    model_fp16 = os.path.join(fp16_dir, f"{base_name}_fp16.onnx")

    # Load model
    model = onnx.load(model_fp32)

    # Convert to FP16
    model_fp16_converted = float16.convert_float_to_float16(model, keep_io_types=True)

    # Save FP16 model
    onnx.save(model_fp16_converted, model_fp16)
    print(f"FP16 model saved to {model_fp16}")
