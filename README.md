# High-Performance ML Inference Engine in C++ using PyTorch and ONNX Runtime


## Overview

This project demonstrates an **end-to-end machine learning deployment pipeline**, starting from **training a CNN in PyTorch** and ending with a **high-performance C++17 inference engine** using **ONNX Runtime**.

The goal is to show **how ML models are actually deployed in production**, with:

* clean data handling
* reproducible training
* ONNX export & compatibility handling
* native C++ inference
* fair Python vs C++ benchmarking

The project is **CPU-only**, portable, and reproducible.

---

## Key Features

* ✔ PyTorch CNN trained on CIFAR-10
* ✔ Exploratory Data Analysis (EDA)
* ✔ Evaluation on real test data
* ✔ ONNX export with IR compatibility handling
* ✔ C++17 inference engine using ONNX Runtime
* ✔ Python vs C++ latency benchmarking
* ✔ Clean project structure & reproducible setup

---

## Project Structure

```
cifar_cpp_inference/
├── python/
│   ├── train_and_export.ipynb
│   ├── visualize_steps.ipynb
│   ├── benchmark_python.ipynb
│   └── model.py
│
├── cpp/
│   ├── main.cpp
│   ├── inference_engine.hpp
│   ├── inference_engine.cpp
│   └── CMakeLists.txt
│
├── models/
│   ├── cifar_tinynet.pth
│   └── cifar_tinynet.onnx
│
├── data/
│   └── (CIFAR-10 dataset – ignored in git)
│
├── results/
│   ├── confusion_matrix.png
│   └── latency_comparison.png
│
├── pip_requirements.txt
├── .gitignore
└── README.md
```

---

## Environment Setup (Python)

### Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python dependencies

```bash
pip install --upgrade pip
pip install -r pip_requirements.txt
```

---

## Dataset Handling (CIFAR-10)

* Dataset is **automatically downloaded only once**
* Subsequent runs reuse the local copy
* Dataset is **ignored by git** to keep repository lightweight

Expected folder after first run:

```
data/
└── cifar-10-batches-py/
```

---

## Model Training, EDA & Export (Python)

### Run training & export notebook

```text
python/train_and_export.ipynb
```

This notebook performs:

* Dataset existence check
* Exploratory Data Analysis (EDA)
* CNN training (`CifarTinyNet`)
* Evaluation on real CIFAR-10 test set
* Confusion matrix & classification report
* Saving:

  * `models/cifar_tinynet.pth`
  * `models/cifar_tinynet.onnx`

### ⚠ Important: ONNX IR Downgrade

After ONNX export, the notebook **downgrades the ONNX IR version** to ensure compatibility with the C++ ONNX Runtime.

This is required when using **modern PyTorch with older runtimes**.

---

## Model Visualization (Python)

### Run visualization notebook

```text
python/visualize_steps.ipynb
```

This notebook:

* Loads the trained model
* Visualizes:

  * raw CIFAR-10 images
  * CNN feature maps
  * logits and softmax confidence
* Performs **no training**
* Does **not download the dataset again**

---

## C++ Inference Engine Setup

### Install ONNX Runtime (C++)

#### Recommended (official prebuilt, CPU-only):

```bash
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar -xvf onnxruntime-linux-x64-1.17.1.tgz
sudo mv onnxruntime-linux-x64-1.17.1 /opt/onnxruntime
```

If required, add runtime library path:

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
```

---

### Build C++ inference engine

```bash
cd cpp
cmake -S . -B build
cmake --build build
```

This builds the binary:

```
cpp/build/cifar_infer
```

---

### Run C++ inference

```bash
./build/cifar_infer
```

Example output:

```
C++ Avg Latency (steady-state): 0.08 ms
```

---

## Python vs C++ Benchmarking

### Run benchmarking notebook

```text
python/benchmark_python.ipynb
```

This notebook:

* Uses **10 real CIFAR-10 test images**
* Benchmarks:

  * Python (PyTorch) inference
  * C++ inference (called via subprocess)
* Uses warm-up + steady-state timing
* Plots latency comparison
* Saves results to `results/`

---

## Benchmark Interpretation

* Python inference is competitive for very small models
* C++ inference scales better and shows lower steady-state latency
* Results demonstrate **realistic ML systems performance trade-offs**

---

## Technologies Used

* **Python**: PyTorch, NumPy, scikit-learn, matplotlib
* **Model Format**: ONNX
* **C++**: C++17
* **Runtime**: ONNX Runtime
* **Build System**: CMake
* **Dataset**: CIFAR-10

---

## Developer

**Mihir Patel**
M.Tech (Computer Science & Engineering)
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat

---

## Key Learning Outcomes

* End-to-end ML deployment workflow
* ONNX export & runtime compatibility handling
* Native C++ inference integration
* Fair benchmarking methodology
* ML + systems engineering best practices

---

