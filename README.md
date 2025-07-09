# EdgeVision
Deep learning pipeline for industrial surface defect detection using TensorFlow MobileNetV2. Loads DAGM and MVTec datasets, classifies images as defective or not, and identifies errors for automated quality control and manufacturing inspection.

---

## Features

- Loads and preprocesses **DAGM** and **MVTec** industrial defect datasets
- Uses **MobileNetV2** for feature extraction and classification
- Computes **precision and recall** metrics for evaluation
- Includes robust logging for traceability
- Modular, scalable, and research-ready

---

## Datasets

**DAGM dataset (DAGM 2007 Classification Benchmark)**
- [Download here (Kaggle)](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)  

**MVTec Anomaly Detection dataset**
- [Download here](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)

---

## ⚙Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/EdgeVision.git
cd EdgeVision
