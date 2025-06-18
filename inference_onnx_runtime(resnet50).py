import onnxruntime as ort
import numpy as np
import os
import time
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
from tqdm import tqdm  # for progress bar

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\test'
onnx_model_path = "resnet50_fruits.onnx"
num_repeats = 100

# === Setup ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

# Collect all image paths and labels
image_paths = []
image_labels = []
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_file))
            image_labels.append(class_name)

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

# === Start total timing ===
total_start_time = time.time()

# Use tqdm progress bar
for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
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
    true_index = class_to_index[true_class]

    y_true.append(true_index)
    y_pred.append(predicted_index)
    total_images += 1
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

# === End total timing ===
total_end_time = time.time()
total_duration = total_end_time - total_start_time

# === Results ===
accuracy = accuracy_score(y_true, y_pred)
avg_inference_time = (total_time_all / total_images) * 1000  # ms

print(f"\n=== ONNX Runtime Inference Report ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_time_all * 1000:.2f} ms")
print(f"Total Inference Duration: {total_duration:.2f} seconds")

print("\nPer-class Performance:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# === Save to TXT ===
txt_report_path = "onnx_inference_report.txt"
with open(txt_report_path, "w") as f:
    f.write("=== ONNX Runtime Inference Report ===\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_time_all * 1000:.2f} ms\n")
    f.write(f"Total Inference Duration: {total_duration:.2f} seconds\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)
print(f"Saved detailed inference report to {txt_report_path}")

# === Save to CSV ===
csv_report_path = "simple_inference_summary_onnx.csv"
summary = {
    "Model": os.path.basename(onnx_model_path),
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_time_all * 1000, 2),
    "Repetitions": num_repeats,
    "Total Time (s)": round(total_duration, 2)
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(csv_report_path, index=False)
print(f"Saved summary to {csv_report_path}")
