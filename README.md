# 🍔 Food Vision - Image Classification with EfficientNetB1

This repository contains an end-to-end **Convolutional Neural Network (CNN)** project that classifies food images into **101 categories** using a **pretrained EfficientNetB1** model fine-tuned on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

---

## 🚀 Project Overview

- Model: EfficientNetB1 (Keras/TensorFlow)
- Dataset: Food-101
- Goal: Classify an input food image into one of 101 categories.
- Training Environment: TensorFlow 2.16 with GPU (supports mixed precision).

---

## 🛠️ Installation & Setup

1. Clone this repository:
```bash
   git clone https://github.com/justaguy1337/food-vision.git
   cd food-vision
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```


3. (Optional) Create a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
```
5. Ensure GPU compatibility (recommended):
   - CUDA 12.x
   - cuDNN matching your TensorFlow version
   - NVIDIA GPU with Compute Capability ≥ 7.0

---

## ▶️ How to Run

### 1. Training
Open the notebook:
   jupyter notebook model_training.ipynb

Run all cells to:
- Load Food-101 dataset
- Preprocess images
- Train EfficientNetB1
- Save the trained model

### 2. Inference
Use the trained model:
  <br><br>
    `from tensorflow.keras.models import load_model`
  <br>
    `model = load_model("saved_models/food_vision_model")`
   # Example prediction
   `pred = model.predict(preprocess_image("example.jpg"))`
   <br>
   `print(pred)`

---

## 📊 Results

- Model: EfficientNetB1
- Dataset: Food-101
- Achieved accuracy: 73%

Example predictions:

Input Image | Predicted Class
------------|---------------
pizza.jpg   | Pizza
sushi.jpg   | Sushi

---

## 📝 Notes

- Uses Mixed Precision Training for speed on compatible GPUs.
- Works well on Google Colab (T4, A100, L4 GPUs, or TPUs).
