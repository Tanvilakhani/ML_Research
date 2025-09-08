import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths to your original FP32 ONNX models
original_models = [
    "Onnx_Models/resnet50_fruits.onnx",
    "Onnx_Models/mobilenetv2_fruits.onnx",
    "Onnx_Models/efficientnetb0_fruits.onnx"
]

# Directory to save quantized models
quantized_dir = "Onnx_Models/quantized"
os.makedirs(quantized_dir, exist_ok=True)  # Create folder if it doesn't exist

# Loop over all models and quantize them dynamically
for model_fp32 in original_models:
    # Extract the base name of the model file without extension
    base_name = os.path.basename(model_fp32).replace(".onnx", "")
    # Construct path for quantized model
    model_int8 = os.path.join(quantized_dir, f"{base_name}_quantized.onnx")
    
    # Perform dynamic quantization (weights only)
    quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)
    
    print(f"Quantized model saved to {model_int8}")
