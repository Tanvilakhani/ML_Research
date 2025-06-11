import onnxruntime as ort
import numpy as np
import os
import time
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications.resnet50 import preprocess_input

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\test'  # Test folder path
onnx_model_path = "resnet50_fruits.onnx"
num_repeats = 100  # Number of inference repetitions per image

# === Setup ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# Load ONNX model
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Tracking
y_true = []
y_pred = []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')

print(f"Running inference {num_repeats} times per image for benchmarking...\n")

# Inference loop
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, img_file)

            # Preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img).astype(np.float32)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Time inference multiple times
            times = []
            for _ in range(num_repeats):
                start_time = time.time()
                pred = session.run([output_name], {input_name: img_array})[0]
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / num_repeats
            min_time = min(times)

            predicted_index = np.argmax(pred)
            true_index = class_to_index[class_name]

            y_true.append(true_index)
            y_pred.append(predicted_index)
            total_images += 1
            total_time_all += avg_time
            min_time_all = min(min_time_all, min_time)

            print(f"Image: {img_file}, True: {class_name}, Predicted: {class_names[predicted_index]}, "
                  f"Avg Time: {avg_time*1000:.2f} ms, Min Time: {min_time*1000:.2f} ms")

# Results
accuracy = accuracy_score(y_true, y_pred)
avg_inference_time = (total_time_all / total_images) * 1000  # ms
print(f"\n=== ONNX Runtime Inference Report ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_time_all * 1000:.2f} ms")

print("\nPer-class Performance:")
print(classification_report(y_true, y_pred, target_names=class_names))
