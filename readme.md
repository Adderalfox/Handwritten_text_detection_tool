📝 Handwritten Character Detection Tool
A deep learning-based tool for detecting and classifying handwritten characters using a custom Convolutional Neural Network (CNN) architecture trained on the EMNIST Balanced dataset.

🚀 Overview
Model: Custom-built CNN with 3 convolutional blocks and fully connected layers

Classes: Supports 62 output classes (digits 0–9, uppercase A–Z, lowercase a–z)

Dataset: EMNIST (Balanced split)

Framework: PyTorch

Purpose: Robust recognition of handwritten characters from grayscale input images

🧠 Model Highlights
Deep CNN with Batch Normalization, ReLU activations, MaxPooling, and Dropout

Final fully connected layers for classification

Designed for grayscale 28x28 input images

🛠️ Training Pipeline
Trained using Cross Entropy Loss and Adam Optimizer

Checkpointing system to resume training from saved states

Final model exported in .pth format

📦 Features
Modular model and training structure

Easy checkpointing and resume support

GPU-compatible for faster training

Ready for integration into OCR pipelines or handwritten form digitization systems

📁 Repository Structure
model.py: CNN architecture

train.py: Training loop and checkpointing logic

utils.py: Helper functions for saving/loading model states

README.md: Project overview

✅ Status
✅ Completed model training
✅ Saved weights
✅ Ready for deployment or inference on custom data
