import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import random

# Import correct preprocessing functions
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# === Model paths ===
model_paths = {
    "resnet50": "Tensorflow_runtime/Models/resnet50_best.h5",
    "mobilenetv2": "Tensorflow_runtime/Models/mobilenetv2_best.h5",
    "efficientnetb0": "Tensorflow_runtime/Models/efficientnetb0_best.h5"
}

# === Output directory for INT8 models ===
save_dir = "Tensorflow_runtime/Models/Int8"
os.makedirs(save_dir, exist_ok=True)

# === Validation data directory ===
VAL_DIR = "MY_data/train"
n_calib_samples = 100  # number of samples for representative dataset

# === Representative dataset generator ===
def representative_data_gen(model_name, n_samples=n_calib_samples):
    img_files = []
    # Collect images from all subfolders
    for cls in os.listdir(VAL_DIR):
        cls_dir = os.path.join(VAL_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_files.append(os.path.join(cls_dir, f))
    # Randomly sample n_samples
    random.shuffle(img_files)
    img_files = img_files[:n_samples]

    for img_path in img_files:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply correct preprocessing
        if model_name == "resnet50":
            img_array = resnet_preprocess(img_array)
        elif model_name == "mobilenetv2":
            img_array = mobilenet_preprocess(img_array)
        elif model_name == "efficientnetb0":
            img_array = efficientnet_preprocess(img_array)

        yield [img_array.astype(np.float32)]

# === Convert models to INT8 TFLite ===
for name, path in model_paths.items():
    print(f"Converting {name} to INT8 TFLite...")
    
    # Load Keras model
    model = tf.keras.models.load_model(path)
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(name)
    
    # Force INT8 I/O
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save converted model
    out_path = os.path.join(save_dir, f"{name}_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Saved {name} INT8 TFLite model at: {out_path}\n")
