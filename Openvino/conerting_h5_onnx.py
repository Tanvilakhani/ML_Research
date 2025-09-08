import tensorflow as tf
import tf2onnx
import os

# Create folder if it doesn't exist
onnx_dir = "Openvino/Models/onnx_models"
os.makedirs(onnx_dir, exist_ok=True)

# List of model files and output ONNX names
models = [
    {"h5": "Tensorflow_runtime/Models/resnet50_best.h5", "onnx": os.path.join(onnx_dir, "resnet50.onnx")},
    {"h5": "Tensorflow_runtime/Models/mobilenetv2_best.h5", "onnx": os.path.join(onnx_dir, "mobilenetv2.onnx")},
    {"h5": "Tensorflow_runtime/Models/efficientnetb0_best.h5", "onnx": os.path.join(onnx_dir, "efficientnetb0.onnx")}
]

for m in models:
    print(f"Converting {m['h5']} to ONNX...")
    
    # Load Keras model (ignore compile to avoid loss issues)
    model = tf.keras.models.load_model(m["h5"], compile=False)
    
    # Use the model's input shape
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    
    # Convert to ONNX
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=m["onnx"])
    
    print(f"ONNX model saved to: {m['onnx']}\n")
