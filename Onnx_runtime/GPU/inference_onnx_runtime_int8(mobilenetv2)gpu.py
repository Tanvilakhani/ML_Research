import os
import time
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import psutil
import gc
import onnxruntime as ort

# === CONFIG ===
IMG_SIZE = 224  # EfficientNetB0 input
IMG_DIR = 'test'

# ---- ONNX Model ----
model_path = 'Onnx_Models/quantized/fp16/resnet50_fruits_int8.onnx'
architecture = "resnet50 (INT8 Quantized)"
runtime = "ONNX Runtime"
num_repeats = 100
device = "GPU"  # Use "CUDAExecutionProvider" for GPU if available

# === OUTPUT PATHS ===
report_dir = "results/report"
os.makedirs(report_dir, exist_ok=True)
txt_report_path = os.path.join(report_dir, f"onnx_{os.path.splitext(os.path.basename(model_path))[0]}_{device}.txt")
csv_report_path = os.path.join("results", "summary_comparison.csv")
os.makedirs("results", exist_ok=True)

# === Load ONNX Model ===
providers = ['CUDAExecutionProvider'] if device.upper() == "GPU" else ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# === Prepare Dataset ===
class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}
index_to_class = {idx: cls_name for cls_name, idx in class_to_index.items()}

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
    img_np = np.array(img).astype(np.float32) / 255.0
    # EfficientNetB0 normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_np, axis=0).astype(np.float16)

# === Inference ===
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')
peak_ram_usage = 0.0

process = psutil.Process(os.getpid())
initial_mem = process.memory_info().rss
peak_total_mem = initial_mem
start_time_all = time.time()

# Warm-up
if image_paths:
    warmup_tensor = preprocess_image(image_paths[0])
    session.run([output_name], {input_name: warmup_tensor})

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    input_tensor = preprocess_image(img_path)
    times = []

    gc.collect()
    mem_before = process.memory_info().rss

    start_time_total = time.time()
    for _ in range(num_repeats):
        t0 = time.time()
        outputs = session.run([output_name], {input_name: input_tensor})
        t1 = time.time()
        times.append(t1 - t0)
    end_time_total = time.time()

    mem_after = process.memory_info().rss
    ram_usage = (mem_after - mem_before) / (1024 ** 2)
    peak_ram_usage = max(peak_ram_usage, ram_usage)

    current_mem = process.memory_info().rss
    peak_total_mem = max(peak_total_mem, current_mem)

    avg_time = (end_time_total - start_time_total) / num_repeats
    min_time = min(times)

    output_tensor = outputs[0]
    predicted_index = int(np.argmax(output_tensor))
    true_index = class_to_index.get(true_class, -1)

    if true_index == -1:
        print(f"Warning: class '{true_class}' not found in mapping. Skipping.")
        continue

    y_true.append(true_index)
    y_pred.append(predicted_index)
    total_images += 1
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

end_time_all = time.time()
total_runtime = end_time_all - start_time_all
total_ram_used = (peak_total_mem - initial_mem) / (1024 ** 2)

# === Metrics ===
accuracy = accuracy_score(y_true, y_pred) if total_images > 0 else 0.0
avg_inference_time = (total_time_all / max(total_images, 1)) * 1000
min_inference_time = (min_time_all if min_time_all != float('inf') else 0.0) * 1000
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

# === Print Report ===
print(f"\n=== ONNX Runtime Inference Report ({architecture}) on {device} ===")
print(f"Model: {os.path.basename(model_path)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms")
print(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB")
print(f"Total Memory Used During Inference (process): {total_ram_used:.2f} MB")
print(f"Total Runtime (All Images): {total_runtime:.2f} seconds\n")
print("Per-class Performance:")
print(report)

# === Save TXT Report ===
with open(txt_report_path, "w") as f:
    f.write(f"=== ONNX Runtime Inference Report ({architecture}) on {device} ===\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n")
    f.write(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB\n")
    f.write(f"Total Memory Used During Inference (process): {total_ram_used:.2f} MB\n")
    f.write(f"Total Runtime (All Images): {total_runtime:.2f} seconds\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)
print(f"Saved detailed inference report to {txt_report_path}")

# === Save to CSV ===
summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(model_path),
    "Architecture": architecture,
    "Format": "ONNX",
    "Runtime": f"{runtime} ({device})",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak RAM Usage (MB)": round(peak_ram_usage, 2),
    "Total Memory Used (MB)": round(total_ram_used, 2),
    "Total Runtime (s)": round(total_runtime, 2),
    "Repetitions": num_repeats,
    "Optimization Option": "None (INT8 Quantized Model)"
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(csv_report_path, index=False)
print(f"Appended summary to {csv_report_path}")
