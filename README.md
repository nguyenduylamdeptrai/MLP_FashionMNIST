# 👕 Fashion-MNIST Classification using Multi-Layer Perceptron (MLP)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Overview
This repository contains a beginner-friendly Deep Learning project that classifies images of clothing from the Fashion-MNIST dataset. Instead of using Convolutional Neural Networks (CNN), this project intentionally uses a traditional **Multi-Layer Perceptron (MLP)** built from scratch using PyTorch to understand the fundamentals of forward propagation, loss calculation, and backpropagation.

## 📂 Project Structure
- `project-1.ipynb`: The Jupyter Notebook containing the data loading, model architecture (Linear layers + ReLU), and the complete training loop with a Cyclic Learning Rate scheduler.
- `main.py`: A standalone Python script used to load the saved model and run inference (predictions) on new/test images.
- `fashion_mlp.pth`: The saved weights of the trained PyTorch model.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install torch torchvision matplotlib numpy
```
### 2. Running the Inference Script
To test the pre-trained model using the Python script, simply run:
```bash
python main.py
```
## 📈 Training Details & Results
- **Dataset:** Fashion-MNIST (60,000 training images, 10,000 test images, 28x28 pixels).
- **Architecture:** - Flatten Layer (28x28 -> 784)
  - Hidden Layer 1 (784 -> 512) + ReLU
  - Hidden Layer 2 (512 -> 256) + ReLU
  - Output Layer (256 -> 10 classes)
- **Optimizer:** SGD with `CyclicLR` (Base LR: 0.001, Max LR: 0.01)
- **Loss Function:** CrossEntropyLoss
- **Final Training Loss:** `0.35`

## 🧠 Key Learnings
Through this project, I have successfully:
- Translated mathematical concepts of Neural Networks into functional PyTorch code.
- Understood the importance of reshaping/flattening image tensors before feeding them into Linear layers.
- Applied **Cyclic Learning Rate (CLR)** to dynamically adjust the learning rate, helping the model escape local minima and converge faster.
- Learned the limitations of MLPs for image processing (loss of spatial information), which serves as a stepping stone for transitioning to CNNs.
