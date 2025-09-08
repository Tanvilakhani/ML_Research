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

from openvino.runtime import Core

# === CONFIG ===
IMG_SIZE = 224  # EfficientNet-B0 default
IMG_DIR = r'MY_data/test'  # Your test image directory
model_xml_path = 'Models/efficientnetb0_fruits_openvino.xml'  # EfficientNet-B0 OpenVINO IR XML model
architecture = "EfficientNet-B0"
format_type = "OpenVINO IR"
runtime = "OpenVINO"
num_repeats = 100
device = "CPU"  # Can also use "GPU" or "AUTO"

# === OUTPUT PATHS ===
report_dir = "results/report"
os.makedirs(report_dir, exist_ok=True)
txt_report_path = os.path.join(report_dir, f"openvino_{os.path.splitext(os.path.basename(model_xml_path))[0]}.txt")
csv_report_path = os.path.join("results", "summary_comparison.csv")
os.makedirs("results", exist_ok=True)

# === Initialize OpenVINO ===
core = Core()
model = core.read_model(model=model_xml_path)

compiled_model = core.compile_model(
    model=model,
    device_name=device,
    config={
        "PERFORMANCE_HINT": "LATENCY"
    }
)

infer_request = compiled_model.create_infer_request()

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

# === Input preprocessing ===
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW
    img_np = np.expand_dims(img_np, axis=0)   # [1, 3, 224, 224]
    return img_np.astype(np.float32)  # ✅ return the tensor


# === Inference on All Images ===
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')
peak_ram_usage = 0.0

process = psutil.Process(os.getpid())
initial_mem = process.memory_info().rss
peak_total_mem = initial_mem
start_time_all = time.time()

print(f"Running inference {num_repeats} times per image using {architecture} ({format_type})...\n")

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    input_tensor = preprocess_image(img_path)

    times = []
    gc.collect()
    mem_before = process.memory_info().rss

    start_time_total = time.time()
    for _ in range(num_repeats):
        start_time = time.time()
        infer_request.infer({compiled_model.input(0): input_tensor})
        end_time = time.time()
        times.append(end_time - start_time)
    end_time_total = time.time()

    mem_after = process.memory_info().rss
    ram_usage = (mem_after - mem_before) / (1024 ** 2)  # MB
    peak_ram_usage = max(peak_ram_usage, ram_usage)

    current_mem = process.memory_info().rss
    peak_total_mem = max(peak_total_mem, current_mem)

    avg_time = (end_time_total - start_time_total) / num_repeats
    min_time = min(times)

    output_tensor = infer_request.get_tensor(compiled_model.output(0)).data
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

accuracy = accuracy_score(y_true, y_pred)
avg_inference_time = (total_time_all / total_images) * 1000
min_inference_time = min_time_all * 1000
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

print(f"\n=== OpenVINO Inference Report ({architecture}) ===")
print(f"Model: {os.path.basename(model_xml_path)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms")
print(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB")
print(f"Total Memory Used During Inference: {total_ram_used:.2f} MB")
print(f"Total Runtime (All Images): {total_runtime:.2f} seconds\n")
print("Per-class Performance:")
print(report)

with open(txt_report_path, "w") as f:
    f.write(f"=== OpenVINO Inference Report ({architecture}) ===\n")
    f.write(f"Model: {os.path.basename(model_xml_path)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n")
    f.write(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB\n")
    f.write(f"Total Memory Used During Inference: {total_ram_used:.2f} MB\n")
    f.write(f"Total Runtime (All Images): {total_runtime:.2f} seconds\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)
print(f"Saved detailed inference report to {txt_report_path}")

summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(model_xml_path),
    "Architecture": architecture,
    "Format": format_type,
    "Runtime": runtime,
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak RAM Usage (MB)": round(peak_ram_usage, 2),
    "Total Memory Used (MB)": round(total_ram_used, 2),
    "Total Runtime (s)": round(total_runtime, 2),
    "Repetitions": num_repeats,
    "Optimization Option": "Graph Optimization"
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(csv_report_path, index=False)
print(f"Appended summary to {csv_report_path}")
