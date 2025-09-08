# import os
# import tf2onnx
# import tensorflow as tf
# import onnx

# # Folder to store ONNX models
# ONNX_DIR = 'Onnx_Models'
# os.makedirs(ONNX_DIR, exist_ok=True)

# # List of models to convert: (keras_model_path, onnx_filename)
# models_to_convert = [
#     ('checkpoints_mobilenetv2/mobilenetv2_best.h5', 'mobilenetv2_fruits(openvino).onnx')
# ]

# # Input spec for ONNX conversion
# input_spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

# # Convert each model
# for keras_path, onnx_filename in models_to_convert:
#     print(f"\nConverting {keras_path} to ONNX...")
    
#     # Load Keras model
#     model = tf.keras.models.load_model(keras_path)
    
#     # Convert to ONNX
#     model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13)
    
#     # Save ONNX model
#     onnx_path = os.path.join(ONNX_DIR, onnx_filename)
#     onnx.save(model_proto, onnx_path)
#     print(f"Saved ONNX model to: {onnx_path}")
    
#     # Validate ONNX model
#     onnx_model = onnx.load(onnx_path)
#     onnx.checker.check_model(onnx_model)
#     print(f"✓ ONNX model check passed: {onnx_filename}")


import os
import tf2onnx
import tensorflow as tf
import onnx

# Folder to store ONNX models
ONNX_DIR = 'Onnx_Models'
os.makedirs(ONNX_DIR, exist_ok=True)

# Models you want to convert
models_to_convert = [
    ('checkpoints_resnet50/resnet50_best.h5', 'resnet50_fruits_openvino.onnx'),
    ('checkpoints_efficientnetb0/efficientnetb0_best.h5', 'efficientnetb0_fruits_openvino.onnx')
]

# TF input signature (NHWC from Keras)
input_spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

for keras_path, onnx_filename in models_to_convert:
    print(f"\nConverting {keras_path} to ONNX in NCHW format...")

    # Load the Keras model
    model = tf.keras.models.load_model(keras_path)

    # Wrap model so ONNX sees NCHW input
    @tf.function(input_signature=(tf.TensorSpec((1, 3, 224, 224), tf.float32, name="input"),))
    def nchw_forward(x):
        # Convert NCHW → NHWC before feeding to Keras model
        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        y = model(x_nhwc)
        return y

    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        nchw_forward,
        input_signature=(tf.TensorSpec((1, 3, 224, 224), tf.float32, name="input"),),
        opset=13
    )

    # Save ONNX
    onnx_path = os.path.join(ONNX_DIR, onnx_filename)
    onnx.save(model_proto, onnx_path)
    print(f"✅ Saved ONNX model to: {onnx_path}")

    # Check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model check passed: {onnx_filename}")
