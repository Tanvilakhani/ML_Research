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
from tensorflow.keras.applications.resnet50 import preprocess_input  

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = 'MY_data/test'
MODEL_PATH = 'Tensorflow_runtime/Models/resnet50_best.h5'
ARCHITECTURE = "ResNet50"
FORMAT_TYPE = "Keras"
RUNTIME = "TensorFlow"
NUM_REPEATS = 100

# GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# CPU affinity
p = psutil.Process(os.getpid())
p.cpu_affinity([0, 1, 2, 3])
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

# === OUTPUT PATHS ===
REPORT_DIR = "results/report"
os.makedirs(REPORT_DIR, exist_ok=True)
TXT_REPORT_PATH = os.path.join(REPORT_DIR, f"optimized_{os.path.splitext(os.path.basename(MODEL_PATH))[0]}.txt")
CSV_REPORT_PATH = os.path.join("results", "summary_comparison.csv")
os.makedirs("results", exist_ok=True)

# === Load Model ===
# === Load Model for Inference Only ===
model = load_model(MODEL_PATH, compile=False)

# === Convert to optimized graph function ===
@tf.function(jit_compile=True)  # Enables XLA optimization
def optimized_inference(x):
    return model(x, training=False)

# === Prepare Dataset ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}

image_paths = []
image_labels = []
for class_name in class_names:
    class_dir = os.path.join(IMG_DIR, class_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_file))
            image_labels.append(class_name)

# === Inference ===
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')
peak_ram_usage = 0.0  # MB
process = psutil.Process(os.getpid())

print(f"Running optimized inference {NUM_REPEATS} times per image on GPU if available...\n")

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        continue

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(preprocess_input(np.array(img).astype(np.float32)), axis=0)
    img_tensor = tf.convert_to_tensor(img_array)

    times = []
    mem_before = process.memory_info().rss

    start_time_total = time.time()
    for _ in range(NUM_REPEATS):
        start_time = time.time()
        pred = optimized_inference(img_tensor)
        end_time = time.time()
        times.append(end_time - start_time)
    end_time_total = time.time()

    mem_after = process.memory_info().rss
    ram_usage = (mem_after - mem_before) / (1024 ** 2)
    peak_ram_usage = max(peak_ram_usage, ram_usage)

    avg_time = (end_time_total - start_time_total) / NUM_REPEATS
    min_time = min(times)

    predicted_index = tf.argmax(pred, axis=1).numpy()[0]
    true_index = class_to_index.get(true_class, -1)
    if true_index == -1:
        print(f"Warning: class '{true_class}' not found. Skipping.")
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

# === Console Output ===
print(f"\n=== Optimized TensorFlow Inference Report ({ARCHITECTURE}) ===")
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms")
print(f"Peak RAM Usage Observed: {peak_ram_usage:.2f} MB\n")
print("Per-class Performance:")
print(report)

# === Save TXT Report ===
with open(TXT_REPORT_PATH, "w") as f:
    f.write(f"=== Optimized TensorFlow Inference Report ({ARCHITECTURE}) ===\n")
    f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n")
    f.write(f"Peak RAM Usage Observed: {peak_ram_usage:.2f} MB\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)
print(f"Saved detailed inference report to {TXT_REPORT_PATH}")

# === Save/Append CSV Summary ===
summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(MODEL_PATH),
    "Architecture": ARCHITECTURE,
    "Format": FORMAT_TYPE,
    "Runtime": RUNTIME,
    "Technique": "Graph Optimization (XLA)",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak RAM Usage (MB)": round(peak_ram_usage, 2),
    "Repetitions": NUM_REPEATS
}

if os.path.exists(CSV_REPORT_PATH):
    df = pd.read_csv(CSV_REPORT_PATH)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(CSV_REPORT_PATH, index=False)
print(f"Appended summary to {CSV_REPORT_PATH}")
