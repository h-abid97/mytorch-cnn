# MyTorch – CNN from Scratch using NumPy

This project is part of the **AI702: Deep Learning** course at MBZUAI (Spring 2024).  
The goal is to build a **Convolutional Neural Network (CNN)** from scratch using **NumPy**, without relying on PyTorch or any deep learning framework.

---

## 📌 Highlights

- 🔨 Implemented custom CNN layers and modules
- 🧮 Manual forward and backward passes using only NumPy
- 🧠 End-to-end training of a CNN model on toy datasets
- ✅ Local autograder support to validate each component

---

## 🧠 Components Implemented

### `mytorch/` – Core Library

- `Conv2D`: 2D convolutional layer (manual sliding + backprop)
- `Flatten`: Flattening layer for transition to FC
- `Linear`: Fully-connected layer with manual gradients
- `ReLU`, `Tanh`, `Sigmoid`: Activation functions
- `CrossEntropyLoss`: Loss computation
- `SGD`: Optimizer with weight updates

### `models/`

- `SimpleCNN`: A small convolutional network with 1–2 conv layers + FC classifier

---

## 📁 Folder Structure

```
cnn-from-scratch-numpy/
├── mytorch/
│ ├── nn/                   # Conv2D, activation, flatten, loss
│ └── optim/                # SGD optimizer
├── models/                 # SimpleCNN model definition
├── autograder/             # Local autograder test scripts
├── MCQ/                    # Optional conceptual questions
├── requirements.txt
└── README.md
```


## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/h-abid97/mytorch-cnn.git
cd mytorch-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Run the appropriate test scripts from the `autograder/` directory:
```bash
python autograder/test_conv2d.py
```

## 📌 Notes
- This project is educational and inspired by PyTorch-style architecture.
- No external deep learning frameworks were used.
