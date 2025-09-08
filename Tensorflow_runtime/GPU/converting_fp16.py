import os
import tensorflow as tf

# === Enable mixed precision for FP16 on GPU ===
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# === Model paths (update if needed) ===
model_paths = {
    "resnet50": "Tensorflow_runtime/Models/resnet50_best.h5",
    "mobilenetv2": "Tensorflow_runtime/Models/mobilenetv2_best.h5",
    "efficientnetb0": "Tensorflow_runtime/Models/efficientnetb0_best.h5"
}

# === Output directory for FP16 models (optional, can just overwrite or save separately) ===
fp16_dir = "Tensorflow_runtime/Models/Fp16"
os.makedirs(fp16_dir, exist_ok=True)

# === Conversion loop (FP16 only) ===
for name, path in model_paths.items():
    print(f"\n=== Loading {name} and converting to FP16 ===")

    # Load Keras model
    model = tf.keras.models.load_model(path, compile=False)

    # Convert model weights to FP16
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            layer.kernel = tf.cast(layer.kernel, dtype=tf.float16)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias = tf.cast(layer.bias, dtype=tf.float16)

    # Save FP16 model
    out_path = os.path.join(fp16_dir, f"{name}_fp16.h5")
    model.save(out_path)

    print(f"Saved FP16 Keras model at: {out_path}")

print("\nAll models converted to FP16 successfully!")
