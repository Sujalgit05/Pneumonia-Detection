A deep learning project that detects pneumonia from chest X-ray images using PyTorch and DenseNet169. The model is fine-tuned on labeled medical datasets to perform binary classification (Pneumonia / Normal) with high accuracy.

📌 Features
Deep Learning Model: Fine-tuned DenseNet169 pre-trained on ImageNet.

Data Preprocessing: Image resizing, normalization, and augmentation using TorchVision and PIL.

GPU Acceleration: Faster training and inference using CUDA.

Real-time Prediction: Sigmoid activation with thresholding for classification.

Batch Prediction: Supports prediction for multiple images at once.

High Accuracy: Achieved 95% classification accuracy.
🛠 Tech Stack
Programming Language: Python

Libraries & Frameworks: PyTorch, TorchVision, PIL, NumPy

Hardware Acceleration: CUDA / GPU

Dataset: Publicly available Chest X-ray dataset
