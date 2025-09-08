import os
import subprocess

# Paths
onnx_dir = "Openvino/Models/onnx_models"
openvino_dir = "Openvino/Models/openvino_models"
models = ["resnet50", "mobilenetv2", "efficientnetb0"]

# Make sure mo is in PATH
mo_cmd = "mo"  # replace with full path to mo.exe if needed

for model in models:
    onnx_path = os.path.join(onnx_dir, f"{model}.onnx")
    
    # FP32 conversion
    fp32_out = os.path.join(openvino_dir, model, "fp32")
    os.makedirs(fp32_out, exist_ok=True)
    print(f"Converting {model} to OpenVINO FP32 IR...")
    subprocess.run([mo_cmd, "--input_model", onnx_path, "--output_dir", fp32_out], check=True)
    
    # FP16 conversion using --compress_to_fp16
    fp16_out = os.path.join(openvino_dir, model, "fp16")
    os.makedirs(fp16_out, exist_ok=True)
    print(f"Converting {model} to OpenVINO FP16 IR...")
    subprocess.run([mo_cmd, "--input_model", onnx_path, "--output_dir", fp16_out, "--compress_to_fp16"], check=True)

print("All models converted to OpenVINO IR (FP32 and FP16) successfully!")
