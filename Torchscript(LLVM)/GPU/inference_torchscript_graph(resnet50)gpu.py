import os
import time
import datetime
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
from torchvision import transforms
import psutil
import gc

# === CONFIG ===
IMG_SIZE = 224
IMG_DIR = 'MY_data/test'
MODEL_PATH = 'Torchscript/Models/resnet50_fruits_scripted.pt'  # change this to your ResNet TorchScript model
ARCHITECTURE = "ResNet"
FORMAT_TYPE = "TorchScript"
RUNTIME = "PyTorch"
NUM_REPEATS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === OUTPUT PATHS ===

REPORT_DIR = "Torchscript/results/report"
os.makedirs(REPORT_DIR, exist_ok=True)
TXT_REPORT_PATH = os.path.join(REPORT_DIR, f"torchscript_graph_{os.path.splitext(os.path.basename(MODEL_PATH))[0]}.txt")
CSV_REPORT_PATH = os.path.join("Torchscript/results", "summary_comparison.csv")
os.makedirs("Torchscript/results", exist_ok=True)

# Enable graph optimization
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

# === Load TorchScript Model ===
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

# Memory tracking
process = psutil.Process(os.getpid())
initial_cpu_mem = process.memory_info().rss
peak_cpu_mem = initial_cpu_mem

if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_time_all = time.time()
print(f"Running inference {NUM_REPEATS} times per image on {DEVICE.upper()} using {ARCHITECTURE} ({FORMAT_TYPE})...\n")

for img_path, true_class in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        continue

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    times = []
    gc.collect()

    start_time_total = time.time()
    for _ in range(NUM_REPEATS):
        start_time = time.time()
        with torch.no_grad():
            pred = model(img_tensor)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    end_time_total = time.time()

    # CPU memory tracking
    current_cpu_mem = process.memory_info().rss
    peak_cpu_mem = max(peak_cpu_mem, current_cpu_mem)

    avg_time = (end_time_total - start_time_total) / NUM_REPEATS
    min_time = min(times)

    predicted_index = torch.argmax(pred, dim=1).item()
    true_index = class_to_index.get(true_class, -1)
    if true_index == -1:
        print(f"Warning: class '{true_class}' not found. Skipping.")
        continue

    y_true.append(true_index)
    y_pred.append(predicted_index)
    total_images += 1
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

# === Final Metrics ===
end_time_all = time.time()
total_runtime = end_time_all - start_time_all

# GPU memory stats
peak_gpu_alloc = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if DEVICE == "cuda" else 0
peak_gpu_reserved = torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 2) if DEVICE == "cuda" else 0
total_gpu_mem_used = peak_gpu_reserved
total_combined_mem = (peak_cpu_mem - initial_cpu_mem) / (1024 ** 2) + total_gpu_mem_used

accuracy = accuracy_score(y_true, y_pred)
avg_inference_time = (total_time_all / total_images) * 1000
min_inference_time = min_time_all * 1000
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

# === Console Output ===
print(f"\n=== TorchScript Inference Report ({ARCHITECTURE}) on {DEVICE.upper()} ===")
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time: {min_inference_time:.2f} ms")
print(f"Peak GPU Memory Allocated: {peak_gpu_alloc:.2f} MB")
print(f"Peak GPU Memory Reserved: {peak_gpu_reserved:.2f} MB")
print(f"Peak CPU Memory: {(peak_cpu_mem - initial_cpu_mem) / (1024 ** 2):.2f} MB")
print(f"Total GPU Memory Used: {total_gpu_mem_used:.2f} MB")
print(f"Total Memory Used (CPU+GPU): {total_combined_mem:.2f} MB")
print(f"Total Runtime: {total_runtime:.2f} seconds")
print("Per-class Performance:")
print(report)

# === Save TXT Report ===
with open(TXT_REPORT_PATH, "w") as f:
    f.write(f"=== TorchScript Inference Report ({ARCHITECTURE}) on {DEVICE.upper()} ===\n")
    f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time: {min_inference_time:.2f} ms\n")
    f.write(f"Peak GPU Memory Allocated: {peak_gpu_alloc:.2f} MB\n")
    f.write(f"Peak GPU Memory Reserved: {peak_gpu_reserved:.2f} MB\n")
    f.write(f"Peak CPU Memory: {(peak_cpu_mem - initial_cpu_mem) / (1024 ** 2):.2f} MB\n")
    f.write(f"Total GPU Memory Used: {total_gpu_mem_used:.2f} MB\n")
    f.write(f"Total Memory Used (CPU+GPU): {total_combined_mem:.2f} MB\n")
    f.write(f"Total Runtime: {total_runtime:.2f} seconds\n\n")
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
    "Technique": "Graph Optimization",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak GPU Alloc (MB)": round(peak_gpu_alloc, 2),
    "Peak GPU Reserved (MB)": round(peak_gpu_reserved, 2),
    "Peak CPU Mem (MB)": round((peak_cpu_mem - initial_cpu_mem) / (1024 ** 2), 2),
    "Total GPU Mem Used (MB)": round(total_gpu_mem_used, 2),
    "Total Mem Used (MB)": round(total_combined_mem, 2),
    "Total Runtime (s)": round(total_runtime, 2),
    "Repetitions": NUM_REPEATS
}

if os.path.exists(CSV_REPORT_PATH):
    df = pd.read_csv(CSV_REPORT_PATH)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])

df.to_csv(CSV_REPORT_PATH, index=False)
print(f"Appended summary to {CSV_REPORT_PATH}")
