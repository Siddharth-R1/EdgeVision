# 🦾 EdgeVision – Advanced Defect Detection with MobileNetV2, DAGM, and MVTec

> **Author:** Siddharth Ramachandran  
> **Domain:** Industrial Computer Vision • AI/ML • Model Optimization for Edge Devices

---

## 🚀 **Project Overview**

EdgeVision is a **robust, scalable defect detection pipeline** integrating:

- 🔬 **DAGM & MVTec industrial defect datasets**
- 🤖 **MobileNetV2 deep learning architecture** with:
  - Advanced data augmentation
  - Focal loss for class imbalance
  - Aggressive oversampling strategies
- ⚡ **Saliency-based defect localisation**
- 📈 **Threshold tuning** for precision-critical applications
- 🌐 **Quantization-aware TFLite conversion** for deployment on **Raspberry Pi and Edge devices**

This project exemplifies a **production-grade AI pipeline** for real-world defect detection in manufacturing and inspection systems.

---

## 🗂️ **Dataset**

- **DAGM Dataset**  
  - 6 industrial defect classes  
  - Loaded with label parsing and metadata handling

- **MVTec Anomaly Detection Dataset**  
  - Category-wise structured loading  
  - Integrated train-test splits with stratification

---

## 🛠️ **Key Features & Techniques**

✅ **Preprocessing & Loading**
- Efficient metadata parsing  
- Preloaded image tensors with Keras pipelines  
- Dynamic sample selection for rapid experimentation

✅ **Model Architecture**
- **MobileNetV2 backbone** with:
  - Custom dense layers
  - Batch normalization and dropout regularization
  - Random flip, rotation, zoom, and contrast augmentations

✅ **Loss & Metrics**
- **Focal Loss** with adaptive alpha for defective class emphasis  
- **Custom F1-Score metric** with Precision and Recall tracking

✅ **Training Strategy**
- Cosine decay learning rate scheduler with warmup  
- Aggressive oversampling of defective samples with image flips  
- Early stopping and model checkpointing on validation F1-Score

✅ **Evaluation & Threshold Tuning**
- Multi-threshold evaluation loop (0.1 to 0.5)
- Per-threshold Precision, Recall, F1-Score, and Confusion Matrix breakdown

✅ **Defect Localization**
- **Saliency map generation** via gradients
- Otsu thresholding with contour-based defect region extraction
- **Fallback detection** using intensity-based difference maps for robustness

✅ **Edge Deployment**
- Full model conversion to **TFLite with int8 quantization**  
- Optimized for Raspberry Pi inference workloads

---

## 🎯 **Sample Outputs**

### ✨ **Defect Localization Example**

| Original Image | Saliency Map | Localized Defect |
|---|---|---|
| (Images/Output1.png) | (Images/Output2.png) | (Images/Output3.png) |

---

## 💻 **Usage**

1. **Clone the repo**

```bash
git clone https://github.com/Siddharth-R1/EdgeVision.git
cd EdgeVision
