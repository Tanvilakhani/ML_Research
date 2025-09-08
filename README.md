# Evaluating the Performance of Different Machine Learning Optimization Frameworks

This repository contains the MSc Artificial Intelligence dissertation project (University of Plymouth, PROJ518).
This repository provides a modular pipeline for training, converting, and benchmarking deep learning models across different optimization frameworks. The goal is to evaluate how frameworks like ONNX Runtime, OpenVINO, TensorFlow Lite, and TorchScript impact model performance.

---

## 📖 Project Overview
This repository contains modular Python scripts for:
- Training of popular CNN architectures: EfficientNetB0, MobileNetV2, ResNet50
- Model Conversion to compatible frameworks styl
- Inference using Keras, ONNX Runtime, TensorFlow Lite, TorchScript, and OpenVINO
- Data Balancing for training and testing splits
- Performance Evaluation with standardized metrics and result logging

---

## 📊 Dataset

This project uses the **Fruit Classification Dataset** containing **3,374 images** across **10 fruit classes**:

- 🍎 Apple  
- 🍊 Orange  
- 🥑 Avocado  
- 🥝 Kiwi  
- 🥭 Mango  
- 🍍 Pineapple  
- 🍓 Strawberries  
- 🍌 Banana  
- 🍒 Cherry  
- 🍉 Watermelon  

👉 [Download the dataset here](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)
 
---

## 📂 Repository Structure
```
├── Converting_models/ # Scripts for converting trained models
├── Onnx_runtime/ # Inference & evaluation using ONNX Runtime
├── Openvino/ # Inference & evaluation using OpenVINO
├── TFlite/ # Inference & evaluation using TensorFlow Lite
├── Tensorflow_runtime/ # Inference & evaluation with native TensorFlow
├── Torchscript(LLVM)/ # Inference & evaluation using TorchScript
├── Training_scripts/ # Training scripts for CNN models
├── results/ # Benchmark results and evaluation outputs
└── README.md # Project documentation

 

**## Future Directions**
- Extend experiments to architectures like EfficientNetV2, ConvNeXt, and Vision Transformers.  
- Test across diverse hardware platforms (ARM, Jetson, Apple M-series).  
- Analyze energy efficiency and performance-per-watt.  
- Explore mixed-precision and quantization-aware training techniques.  
- Benchmark inference in real-time and batch workloads.  

