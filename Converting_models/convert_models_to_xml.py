from openvino import convert_model, serialize
from pathlib import Path

models = {
    "resnet50_fruits_openvino": "Onnx_Models/resnet50_fruits_openvino.onnx",
    # "mobilenet_v2_fruits_openvino": "Onnx_Models/mobilenetv2_fruits(openvino).onnx",
    "efficientnetb0_fruits_openvino": "Onnx_Models/efficientnetb0_fruits_openvino.onnx"
}

output_dir = Path("Models")
output_dir.mkdir(exist_ok=True)

for model_name, onnx_path in models.items():
    print(f"Converting {model_name}...")

    # Convert without input_shape argument
    model_ir = convert_model(onnx_path)

    xml_path = output_dir / f"{model_name}.xml"
    bin_path = output_dir / f"{model_name}.bin"
    serialize(model_ir, xml_path, bin_path)

    print(f"✅ Saved {model_name} to {xml_path}\n")

print("All models converted successfully.")
