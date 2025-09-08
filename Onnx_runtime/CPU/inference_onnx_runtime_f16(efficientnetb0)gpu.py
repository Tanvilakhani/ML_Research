# -------------------------
# IMPORTS
# -------------------------
import os
import time
import datetime
import psutil
import gc
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import torch
import onnxruntime as ort
from torchvision import transforms
from tqdm import tqdm

# -------------------------
# CONFIGURATION
# -------------------------
IMG_SIZE = 224
IMG_DIR = '/content/drive/MyDrive/test'
model_path = '/content/drive/MyDrive/fp16/efficientnetb0_fruits_fp16.onnx'
architecture = "Efficientnetb0"
format_type = "ONNX"
runtime_name = "ONNX Runtime"
num_repeats = 100
device = "CUDA" if torch.cuda.is_available() else "CPU"
batch_size = 1

# OUTPUT PATHS
# Directory for reports
report_dir = "/content/drive/MyDrive/Results/Reports"
os.makedirs(report_dir, exist_ok=True)
txt_report_path = os.path.join(
    report_dir,
    f"onnx_{os.path.splitext(os.path.basename(model_path))[0]}_{device}.txt"
)
csv_report_path = "/content/drive/MyDrive/Results/summary_comparison.csv"
os.makedirs(os.path.dirname(csv_report_path), exist_ok=True)

print(f"Using device: {device}")


# -------------------------
# MEMORY UTILITIES
# -------------------------
def get_cpu_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

# -------------------------
# DATA PREPROCESSING
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = sorted([d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))])
class_to_index = {cls_name: idx for idx, cls_name in enumerate(class_names)}
index_to_class = {idx: cls_name for cls_name, idx in class_to_index.items()}

image_paths, image_labels = [], []
for cls_name in class_names:
    class_dir = os.path.join(IMG_DIR, cls_name)
    for img_file in os.listdir(class_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, img_file))
            image_labels.append(class_to_index[cls_name])

print(f"Found {len(image_paths)} images across {len(class_names)} classes.")

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).numpy().astype(np.float16)  # <-- cast to float16
    return img_tensor

# -------------------------
# LOAD ONNX MODEL
# -------------------------
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device=="CUDA" else ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_type = session.get_inputs()[0].type
print(f"ONNX model expects input type: {input_type}")


# -------------------------
# INFERENCE
# -------------------------
y_true, y_pred = [], []
total_images = 0
total_time_all = 0.0
min_time_all = float('inf')
peak_ram_usage = 0.0

process = psutil.Process(os.getpid())
initial_mem = process.memory_info().rss
peak_total_mem = initial_mem

start_time_all = time.time()
print(f"\nRunning inference {num_repeats} times per image using {architecture} ({format_type}) on {device}...\n")

for img_path, true_label in tqdm(zip(image_paths, image_labels), total=len(image_paths), desc="Processing Images"):
    input_tensor = preprocess_image(img_path)
    times = []

    gc.collect()
    mem_before = process.memory_info().rss

    # Repeat inference per image
    for _ in range(num_repeats):
        t0 = time.time()
        batch_output = session.run([output_name], {input_name: input_tensor})[0]
        t1 = time.time()
        times.append(t1 - t0)

    mem_after = process.memory_info().rss
    ram_usage = (mem_after - mem_before) / (1024 ** 2)
    peak_ram_usage = max(peak_ram_usage, ram_usage)
    peak_total_mem = max(peak_total_mem, mem_after)

    avg_time = np.mean(times)
    min_time = np.min(times)
    total_time_all += avg_time
    min_time_all = min(min_time_all, min_time)

    pred_index = int(np.argmax(batch_output))
    y_true.append(true_label)
    y_pred.append(pred_index)
    total_images += 1

end_time_all = time.time()
total_runtime = end_time_all - start_time_all
total_mem_used = (peak_total_mem - initial_mem) / (1024 ** 2)

# -------------------------
# METRICS
# -------------------------
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
avg_inference_time = (total_time_all / max(total_images, 1)) * 1000  # ms
min_inference_time = min_time_all * 1000  # ms

# -------------------------
# PRINT REPORT
# -------------------------
print(f"\n=== ONNX Runtime Inference Report ({architecture}) on {device} ===")
print(f"Model: {os.path.basename(model_path)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average Inference Time per Image: {avg_inference_time:.2f} ms")
print(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms")
print(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB")
print(f"Total Memory Used During Inference: {total_mem_used:.2f} MB")
print(f"Total Runtime (All Images): {total_runtime:.2f} s\n")
print("Per-class Performance:")
print(report)

# -------------------------
# SAVE TXT REPORT
# -------------------------
with open(txt_report_path, "w") as f:
    f.write(f"=== ONNX Runtime Inference Report ({architecture}) on {device} ===\n")
    f.write(f"Model: {os.path.basename(model_path)}\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Average Inference Time per Image: {avg_inference_time:.2f} ms\n")
    f.write(f"Minimum Inference Time Observed: {min_inference_time:.2f} ms\n")
    f.write(f"Peak RAM Usage Observed (per image): {peak_ram_usage:.2f} MB\n")
    f.write(f"Total Memory Used During Inference: {total_mem_used:.2f} MB\n")
    f.write(f"Total Runtime (All Images): {total_runtime:.2f} s\n\n")
    f.write("Per-class Performance:\n")
    f.write(report)

# -------------------------
# SAVE CSV SUMMARY
# -------------------------
summary = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": os.path.basename(model_path),
    "Architecture": architecture,
    "Format": format_type,
    "Runtime": f"{runtime_name} ({device})",
    "Accuracy (%)": round(accuracy * 100, 2),
    "Avg Inference Time (ms)": round(avg_inference_time, 2),
    "Min Inference Time (ms)": round(min_inference_time, 2),
    "Peak RAM Usage (MB)": round(peak_ram_usage, 2),
    "Total Memory Used (MB)": round(total_mem_used, 2),
    "Total Runtime (s)": round(total_runtime, 2),
    "Repetitions": num_repeats,
    "Optimization Option": "FP16"
}

if os.path.exists(csv_report_path):
    df = pd.read_csv(csv_report_path)
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
else:
    df = pd.DataFrame([summary])
df.to_csv(csv_report_path, index=False)

print(f"\nTXT report saved to {txt_report_path}")
print(f"CSV summary saved to {csv_report_path}")
