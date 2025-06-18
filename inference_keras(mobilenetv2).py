import os
import time
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = r'MY_data/test'  # your test set path
model_path = 'checkpoints_mobilenetv2/mobilenetv2_best.h5' 
architecture = "MobileNetV2"
format_type = "Keras"
runtime = "TensorFlow"
num_repeats = 100

# === OUTPUT PATHS ===
report_dir = "results/report"
os.makedirs(report_dir, exist_ok=True)
txt_report_path = os.path.join(report_dir, f"keras_{os.path.splitext(os.path.basename(model_path))[0]}.txt")
csv_report_path = os.path.join("results", "summary_comparison.csv")
os.makedirs("results", exist_ok=True)

# === Load Model ===
model = load_model(model_path)

# === Prepare Dataset ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}
index_to_class = {idx: cls_name for cls_name, idx in class_to_index.items()}

image_paths = []
image_labels = []
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_file))
            image_labels.append(class_name)

# === Inference on All Images ===
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')

print(f"Running inference {num_repeats} times per image using {architecture} ({format_type})...\n")

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        continue

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(preprocess_input(np.array(img).astype(np.float32)), axis=0)

    times = []
    start_time_total = time.time()
    for _ in range(num_repeats):
        start_time = time.time()
        pred = model.predict(img_array, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    end_time_total = time.time()

    avg_time = (end_time_total - start_time_total) / num_repeats
    min_time = min(times)

    predicted_index = np.argmax(pred)
    true_index = class_to_index.get(true_class, -1)

    if true_index == -1:
        print(f"Warning: class '{true_class}' not found in mapping. Skipping.")
        continue

    y_true.append(true_index)
    y_pred.append(predicted_index)
    total_images += 1
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

# === Metrics ===
accuracy = accuracy_score(y_true, y_pred)
avg_inference_time = (total_time_all / total_images) * 1000  # ms
min_inference_time = min_time_all * 1000  # ms
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

# === Print metrics to console ===
print(f"\n=== Keras Inference Report ({architecture}) ===")
print(f"Model: {os.path.basename(model_path)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n")
print("Per-class Performance:")
print(report)

# === Save TXT Report ===
with open(txt_report_path, "w") as f:
    f.write(f"=== Keras Inference Report ({architecture}) ===\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)
print(f"Saved detailed inference report to {txt_report_path}")

# === Save to CSV ===
summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(model_path),
    "Architecture": architecture,
    "Format": format_type,
    "Runtime": runtime,
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Repetitions": num_repeats
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(csv_report_path, index=False)
print(f"Appended summary to {csv_report_path}")
