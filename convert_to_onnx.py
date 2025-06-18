import os
import tf2onnx
import tensorflow as tf
import onnx

# Folder to store ONNX models
ONNX_DIR = 'Onnx_Models'
os.makedirs(ONNX_DIR, exist_ok=True)

# List of models to convert: (keras_model_path, onnx_filename)
models_to_convert = [
    ('checkpoints_resnet50/resnet50_best.h5', 'resnet50_fruits.onnx'),
    ('checkpoints_mobilenetv2/mobilenetv2_best.h5', 'mobilenetv2_fruits.onnx'),
    ('checkpoints_efficientnetb0/efficientnetb0_best.h5', 'efficientnetb0_fruits.onnx')
]

# Input spec for ONNX conversion
input_spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convert each model
for keras_path, onnx_filename in models_to_convert:
    print(f"\nConverting {keras_path} to ONNX...")
    
    # Load Keras model
    model = tf.keras.models.load_model(keras_path)
    
    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13)
    
    # Save ONNX model
    onnx_path = os.path.join(ONNX_DIR, onnx_filename)
    onnx.save(model_proto, onnx_path)
    print(f"Saved ONNX model to: {onnx_path}")
    
    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model check passed: {onnx_filename}")
