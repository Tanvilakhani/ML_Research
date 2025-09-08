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


