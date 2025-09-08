# Evaluating the Performance of Different ML Optimization Frameworks

This repository provides a modular pipeline for training, converting, and benchmarking deep learning models across different optimization frameworks. The goal is to evaluate how frameworks like ONNX Runtime, OpenVINO, TensorFlow Lite, and TorchScript impact model performance.

---

## 📖 Project Overview
- Train CNN models (*EfficientNetB0, MobileNetV2, ResNet50*).  
- Convert trained models into multiple formats (ONNX, TFLite, TorchScript, OpenVINO).  
- Benchmark models on accuracy, inference latency, throughput, and memory usage.  
- Analyze trade-offs between optimization efficiency and predictive fidelity.  

---

## 📊 Dataset
**Fruit Classification Dataset – 10 Classes**  
👉 [Dataset link](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)

- Preprocessing: resize (224×224), normalization, random augmentations.  
- Train/test split: 80% training, 20% testing.  

---

## 🖥️ Experimental Environment
- **Hardware:**
  - CPU: Intel i7 / AMD Ryzen (replace with your actual specs)  
  - GPU: NVIDIA RTX / GTX (replace with your GPU)  
  - RAM: 16 GB  

- **OS:** Ubuntu 22.04 LTS  
- **Python:** 3.10+  

- **Frameworks & Backends:**
  - TensorFlow 2.x  
  - PyTorch 2.x  
  - ONNX Runtime  
  - OpenVINO Toolkit  
  - TensorFlow Lite  
  - TorchScript  

---

## 📂 Repository Structure

Evaluating the Performance of Different ML Optimization Frameworks

This repository provides a modular pipeline for training, converting, and benchmarking deep learning models across different optimization frameworks. The goal is to evaluate how frameworks like ONNX Runtime, OpenVINO, TensorFlow Lite, and TorchScript impact model performance.

This repository contains modular Python scripts for:

🚀 Features

Training of popular CNN architectures: EfficientNetB0, MobileNetV2, ResNet50
Model Conversion to compatible frameworks styl
Inference using Keras, ONNX Runtime, TensorFlow Lite, TorchScript, and OpenVINO
Data Balancing for training and testing splits
Performance Evaluation with standardized metrics and result logging

📂 Repository Structure

Converting_models/       # Scripts for converting trained models
Onnx_runtime/            # Inference & evaluation using ONNX Runtime
Openvino/                # Inference & evaluation using OpenVINO
TFlite/                  # Inference & evaluation using TensorFlow Lite
Tensorflow_runtime/      # Inference & evaluation with native TensorFlow
Torchscript(LLVM)/       # Inference & evaluation using TorchScript
Training_scripts/        # Training scripts for CNN models
results/                 # Benchmark results and evaluation outputs
README.md                # Project documentation

Dataset


Download the dataset used in this project here:  
CNN Task: Fruit classification https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data


