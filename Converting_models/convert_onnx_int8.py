import os
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

# Paths to your original FP32 ONNX models
original_models = [
    "Onnx_Models/resnet50_fruits.onnx",
    "Onnx_Models/mobilenetv2_fruits.onnx",
    "Onnx_Models/efficientnetb0_fruits.onnx"
]

# Directory to save INT8 quantized models
int8_dir = "Onnx_Models/int8"
os.makedirs(int8_dir, exist_ok=True)

# -------------------------
# DYNAMIC QUANTIZATION (simpler, no calibration dataset needed)
# -------------------------
for model_fp32 in original_models:
    base_name = os.path.basename(model_fp32).replace(".onnx", "")
    model_int8 = os.path.join(int8_dir, f"{base_name}_int8.onnx")
    
    # Dynamic quantization: weights quantized to INT8
    quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)
    print(f"Dynamic INT8 model saved to {model_int8}")


