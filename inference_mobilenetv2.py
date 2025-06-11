import onnxruntime as ort
import numpy as np
import os
import time
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set parameters
IMG_SIZE = 224
IMG_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\test'  # Test folder path
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# Load ONNX model
onnx_model_path = "mobilenetv2_fruits.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Initialize tracking
y_true = []
y_pred = []
total_images = 0
total_time = 0.0

# Loop over test images
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, img_file)

            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img).astype(np.float32)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Run inference
            start_time = time.time()
            pred = session.run([output_name], {input_name: img_array})[0]
            end_time = time.time()
            total_time += (end_time - start_time)

            predicted_index = np.argmax(pred)
            true_index = class_to_index[class_name]

            y_true.append(true_index)
            y_pred.append(predicted_index)
            total_images += 1

            # Print prediction for each image
            print(f"Image: {img_file}, True: {class_name}, Predicted: {class_names[predicted_index]}")

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
avg_time_ms = (total_time / total_images) * 1000

# Print performance metrics
print("\nMobileNetV2 ONNX Runtime Inference Results:")
print("-" * 45)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Total Inference Time: {total_time:.2f} seconds")
print(f"Average Inference Time per Image: {avg_time_ms:.2f} ms")
print(f"Total Images Processed: {total_images}")

# Print detailed per-class performance
print("\nPer-class Performance:")
print("-" * 45)
print(classification_report(y_true, y_pred, target_names=class_names))