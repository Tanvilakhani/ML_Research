import os
import time
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import psutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = 'MY_data/test'
MODEL_PATH = 'Tensorflow_runtime/Models/int8/efficientnetb0_int8.h5' 
architecture = "efficientnetb0"
format_type = "Keras"
runtime = "TensorFlow"
num_repeats = 100
device = "GPU"

# === GPU SETTINGS ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === OUTPUT PATHS ===
report_dir = "Tensorflow/results/report"
os.makedirs(report_dir, exist_ok=True)
txt_report_path = os.path.join(report_dir, f"tensorflow_{os.path.splitext(os.path.basename(MODEL_PATH))[0]}_{device}.txt")
csv_report_path = os.path.join("Tensorflow/results", "summary_comparison.csv")
os.makedirs("Tensorflow/results", exist_ok=True)

# === Load FP16 Model ===
model = load_model(MODEL_PATH, compile=False)

# === Dataset Preparation ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

image_paths, image_labels = [], []
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_file))
            image_labels.append(class_name)

# === Preprocessing ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32)
    img_np = preprocess_input(img_np)  # MobileNetV2 preprocessing
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

# === Inference ===
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')
peak_ram_usage = 0.0
process = psutil.Process(os.getpid())
initial_cpu_mem = process.memory_info().rss
peak_total_mem = initial_cpu_mem

start_time_all = time.time()

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    input_tensor = tf.convert_to_tensor(preprocess_image(img_path), dtype=tf.float16)
    times = []
    mem_before = process.memory_info().rss

    start_time_total = time.time()
    for _ in range(num_repeats):
        t0 = time.time()
        output = model(input_tensor, training=False)  # FP16 inference
        t1 = time.time()
        times.append(t1 - t0)
    end_time_total = time.time()

    mem_after = process.memory_info().rss
    ram_usage = (mem_after - mem_before) / (1024 ** 2)
    peak_ram_usage = max(peak_ram_usage, ram_usage)
    peak_total_mem = max(peak_total_mem, process.memory_info().rss)

    avg_time = (end_time_total - start_time_total) / num_repeats
    min_time = min(times)

    predicted_index = int(tf.argmax(output, axis=1).numpy()[0])
    true_index = class_to_index.get(true_class, -1)
    if true_index == -1:
        continue

    y_true.append(true_index)
    y_pred.append(predicted_index)
    total_images += 1
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

end_time_all = time.time()
total_runtime = end_time_all - start_time_all
total_cpu_mem_used = (peak_total_mem - initial_cpu_mem) / (1024 ** 2)

# === GPU memory tracking ===
total_gpu_mem_used = 0.0
if gpus and device == "GPU":
    try:
        gpu_mem_info = tf.config.experimental.get_memory_info('GPU:0')
        total_gpu_mem_used = gpu_mem_info['peak'] / (1024 ** 2)
    except:
        total_gpu_mem_used = 0.0

total_combined_mem = total_cpu_mem_used + total_gpu_mem_used

# === Metrics ===
accuracy = accuracy_score(y_true, y_pred) if total_images > 0 else 0.0
avg_inference_time = (total_time_all / max(total_images, 1)) * 1000
min_inference_time = (min_time_all if min_time_all != float('inf') else 0.0) * 1000
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

# === Print Report ===
print(f"\n=== TensorFlow FP16 Inference Report ({architecture}) on {device} ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Avg Inference Time: {avg_inference_time:.2f} ms")
print(f"Min Inference Time: {min_inference_time:.2f} ms")
print(f"Peak RAM Usage: {peak_ram_usage:.2f} MB")
print(f"Total GPU Mem Used: {total_gpu_mem_used:.2f} MB")
print(f"Total Memory Used (CPU+GPU): {total_combined_mem:.2f} MB")
print(report)

# === Save Reports ===
with open(txt_report_path, "w") as f:
    f.write(report)

summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(MODEL_PATH),
    "Architecture": architecture,
    "Format": format_type,
    "Runtime": f"{runtime} ({device})",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak RAM Usage (MB)": round(peak_ram_usage, 2),
    "Total GPU Mem Used (MB)": round(total_gpu_mem_used, 2),
    "Total Mem Used (MB)": round(total_combined_mem, 2),
    "Repetitions": num_repeats,
    "Optimization Option": "int8 Inference"
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])
df.to_csv(csv_report_path, index=False)
