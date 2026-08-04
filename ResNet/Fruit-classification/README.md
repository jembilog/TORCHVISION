# 🍎 Fruit Classifier using ResNet18 (Transfer Learning)

A deep learning image classification project built with **PyTorch** using **Transfer Learning** with a pretrained **ResNet18** model.

This project classifies **9 different fruit categories** using **Feature Extraction**, where all pretrained convolutional layers are frozen and only the final classification layer is trained.

---

# 📌 Features

- ✅ Transfer Learning using ResNet18
- ✅ Feature Extraction (No Fine-Tuning)
- ✅ PyTorch DataLoader
- ✅ GPU (CUDA) Support
- ✅ Automatic Train / Validation / Test Pipeline
- ✅ Save Best Model
- ✅ Model Evaluation
- ✅ Classification Report
- ✅ Confusion Matrix
- ✅ Single Image Prediction
- ✅ Prediction Confidence
- ✅ Class Probabilities
- ✅ Image Visualization

---

# 🧠 Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
Pretrained ResNet18
(ImageNet Weights)
        │
        ▼
Frozen Convolution Layers
        │
        ▼
Feature Extraction
        │
        ▼
Fully Connected Layer (9 Classes)
        │
        ▼
Prediction
```

---

# 🍓 Fruit Classes

The model classifies the following fruits:

- Apple
- Banana
- Cherry
- Chickoo
- Grapes
- Kiwi
- Mango
- Orange
- Strawberry

---

# 📂 Project Structure

```
fruit-classifier-resnet18/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
│
├── best_model.pth
├── requirements.txt
└── README.md
```

---

# 📁 Dataset Structure

```
dataset/
│
├── train/
│   ├── Apple/
│   ├── Banana/
│   ├── Cherry/
│   ├── Chickoo/
│   ├── Grapes/
│   ├── Kiwi/
│   ├── Mango/
│   ├── Orange/
│   └── Strawberry/
│
├── validation/
│   ├── Apple/
│   └── ...
│
└── test/
    ├── Apple/
    └── ...
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fruit-classifier-resnet18.git
```

Go to the project

```bash
cd fruit-classifier-resnet18
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

Train the model using:

```bash
python train.py
```

The program will:

- Load the dataset
- Train the classifier
- Validate every epoch
- Save the best model automatically

Output:

```
best_model.pth
```

---

# 📊 Evaluation

Run:

```bash
python evaluate.py
```

This script provides:

- Test Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

---

# 🔍 Prediction

Run:

```bash
python predict.py
```

The script displays:

- Actual Class
- Predicted Class
- Confidence Score
- Probabilities for every fruit class
- Input image visualization

Example:

```
Actual Class:
Mango

Predicted Class:
Mango

Confidence:
98.73%

Probabilities

Apple          0.01%
Banana         0.05%
Cherry         0.02%
Chickoo        0.07%
Grapes         0.04%
Kiwi           0.03%
Mango         98.73%
Orange         0.72%
Strawberry     0.33%
```

---

# 🧠 Transfer Learning Strategy

This project uses **Feature Extraction**.

All pretrained ResNet18 convolutional layers remain frozen.

Only the final Fully Connected (FC) layer is trained.

```
Conv1      ❄ Frozen
Layer1     ❄ Frozen
Layer2     ❄ Frozen
Layer3     ❄ Frozen
Layer4     ❄ Frozen
FC Layer   🔥 Trainable
```

This approach is effective for smaller datasets and significantly reduces training time while leveraging ImageNet's learned visual features.

---

# 📈 Learning Objectives

This project demonstrates:

- Image Classification
- Transfer Learning
- Feature Extraction
- Model Training
- Model Evaluation
- Confusion Matrix Analysis
- Prediction Confidence
- Deep Learning Workflow using PyTorch

---

# 📚 Future Improvements

- Fine-Tuning ResNet18
- EfficientNet Implementation
- MobileNet Comparison
- DenseNet Comparison
- Data Augmentation
- Learning Rate Scheduler
- Grad-CAM Visualization
- ONNX Export
- TensorRT Optimization
- Web Deployment with FastAPI
- Mobile Deployment

---

# 👨‍💻 Author

**Jemelrey D. Abastillas**

Computer Engineering Student

Passionate about Artificial Intelligence, Machine Learning, Robotics, and Embedded Systems.

---

# ⭐ Acknowledgements

- PyTorch
- Torchvision
- ImageNet
- Microsoft ResNet Research
