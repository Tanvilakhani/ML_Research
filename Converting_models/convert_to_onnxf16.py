import os
import tensorflow as tf
import tf2onnx
import onnx
from onnxconverter_common import float16

# -------------------------
# CONFIG
# -------------------------
ONNX_DIR = "Onnx_Models"
os.makedirs(ONNX_DIR, exist_ok=True)

# List of models to convert (keras_path, onnx_filename_base)
models_to_convert = [
    ("checkpoints_resnet50/resnet50_best.h5", "resnet50_fruits"),
    ("checkpoints_mobilenetv2/mobilenetv2_best.h5", "mobilenetv2_fruits"),
    ("checkpoints_efficientnetb0/efficientnetb0_best.h5", "efficientnetb0_fruits"),
]

# -------------------------
# CONVERSION LOOP
# -------------------------
for keras_path, onnx_basename in models_to_convert:
    print(f"\nConverting {keras_path} → {onnx_basename}.onnx (NCHW) ...")

    # Load original Keras model (expects NHWC)
    keras_model = tf.keras.models.load_model(keras_path)

    # Wrap it to accept NCHW inputs
    @tf.function(
        input_signature=[tf.TensorSpec((1, 3, 224, 224), tf.float32, name="input")]
    )
    def nchw_forward(x):
        # Convert NCHW → NHWC before feeding into Keras model
        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        y = keras_model(x_nhwc)
        return y

    # Convert function → ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        nchw_forward,
        input_signature=[tf.TensorSpec((1, 3, 224, 224), tf.float32, name="input")],
        opset=13,
    )

    # Save FP32 ONNX model
    onnx_path = os.path.join(ONNX_DIR, f"{onnx_basename}.onnx")
    onnx.save(model_proto, onnx_path)
    print(f"Saved ONNX model to {onnx_path}")

    # Validate FP32 ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX check passed for {onnx_basename}.onnx")

    # -------------------------
    # FP16 Conversion
    # -------------------------
    print(f"Converting {onnx_basename}.onnx → FP16...")
    fp16_model = float16.convert_float_to_float16(onnx_model)

    fp16_path = os.path.join(ONNX_DIR, f"{onnx_basename}_fp16.onnx")
    onnx.save(fp16_model, fp16_path)
    print(f"Saved FP16 model to {fp16_path}")

print("\nAll models converted to NCHW ONNX + FP16 successfully!")
