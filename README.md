# SpamDetectionAI




# 🚀 SMS Spam Detection with C++ Neural Network

**A Production-Ready Neural Network Implementation from Scratch in C++17**

[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.3%25-brightgreen.svg)]()
[![F1 Score](https://img.shields.io/badge/F1%20Score-92.7%25-brightgreen.svg)]()

---

## 📌 Overview

A **high-performance SMS spam detection system** built entirely from scratch in C++17. This project implements a **multi-layer neural network** with backpropagation to classify SMS messages as **spam** or **legitimate (ham)**.

No external machine learning libraries were used. Everything is implemented from the ground up, demonstrating a deep understanding of neural network internals.

### 🎯 Key Achievements

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **98.30%** |
| **Precision** | **96.03%** |
| **Recall** | **89.63%** |
| **F1 Score** | **92.72%** |
| **Train Accuracy** | **99.96%** |
| **Training Time** | **~5.8 minutes** |

### 📊 Confusion Matrix


- ✅ **99.5%** of ham messages correctly classified
- ✅ **89.6%** of spam messages detected
- ✅ Only **5 false alarms** out of 1,115 test messages




### Feature Extraction Pipeline

1. **Tokenization** – Splits text into individual words
2. **Lowercasing** – Normalizes all characters
3. **Punctuation Removal** – Removes non-alphanumeric characters
4. **Bag-of-Words** – Creates a vocabulary of 2,000 most frequent words
5. **Advanced Features** – 5 additional hand-crafted features:

| Feature | Description | Spam Indicator |
|---------|-------------|----------------|
| Upper Case Ratio | Percentage of uppercase letters | Higher in spam |
| Exclamation Count | Presence of "!" | Common in spam |
| Spam Word Score | Spam-related keywords (free, win, prize) | Higher in spam |
| URL Presence | Contains http:// or www | Common in spam |
| Message Length | Normalized text length | Longer messages often spam |






### Training Algorithm

- **Forward Propagation** – Computes predictions through the network
- **Loss Function** – Mean Squared Error (MSE)
- **Backpropagation** – Gradients calculated using chain rule
- **Weight Update** – Stochastic Gradient Descent (SGD) with learning rate = 0.01
- **Weight Initialization** – Xavier/Glorot initialization for optimal convergence
- **Epochs** – 50 complete passes through training data








## 🚀 Getting Started

### Prerequisites

- **Windows:** Visual Studio 2022 with C++ development tools
- **Linux:** GCC 9+ or Clang 10+ with CMake 3.10+

### Installation

#### Windows (Visual Studio 2022)

```bash
# Clone the repository
git clone https://github.com/SedatDurgun/CppAIs.git
cd CppAIs/SpamDetectionAI

# Open Visual Studio solution
open SpamDetectionAI.sln

# Build in Release mode
Build → Configuration Manager → Release
Build → Build Solution

# Run
Debug → Start Without Debugging (Ctrl+F5)
