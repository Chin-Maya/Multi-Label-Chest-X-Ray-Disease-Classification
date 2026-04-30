# Multi-Label Chest X-Ray Disease Classification
Detection and classification up to 14 common thoracic (chest) diseases from a single frontal chest X-ray image using Deep Neural Networks
---

**DenseNet-121** based multi-label classification model for the **NIH ChestX-ray14** dataset using PyTorch.

This repository contains the final cleaned Kaggle notebook that trains a strong multi-label classifier capable of detecting **14 thoracic diseases** from frontal chest X-rays.

---

## 📋 Project Overview

- **Task**: Multi-label classification (one image can have 0 or more diseases)
- **Dataset**: NIH ChestX-ray14 (112,120 frontal chest X-rays)
- **Model**: DenseNet-121 (ImageNet pretrained) + custom classification head
- **Loss**: Binary Focal Loss (handles severe class imbalance)
- **Key Features**:
  - Full training pipeline with mixed precision
  - Comprehensive evaluation (AUC, F1, per-class metrics)
  - Interpretability with Grad-CAM
  - Few-shot learning experiments
  - Demographic bias analysis (gender & age)
  - Ethical considerations discussion

---

## 📁 Repository Structure

```
multi-label-chest-xray-disease-classification/
├── README.md
├── Multi-Label_Chest_XRay_Disease_Classification.ipynb   # Main notebook
├── utils/
│   ├── dataset.py          # Custom Dataset class
│   ├── focal_loss.py       # Focal Loss implementation
│   ├── metrics.py          # Evaluation utilities
│   └── utils.py            # Helper functions (seed, memory, etc.)
├── requirements.txt
└── LICENSE
```

> **Note**: The actual image dataset is **not** included in this repo (too large). You must download it from Kaggle.

---

## 🚀 How to Use

### 1. Run on Kaggle (Recommended)

The easiest way is to use the original Kaggle notebook:

→ **[Open in Kaggle](https://www.kaggle.com/code/chinmayabhargava/multi-label-chest-xray-disease-classification-1)**

It already has the dataset attached and GPU enabled.

### 2. Run Locally / On Your Own Machine

#### Step 1: Download the Dataset

1. Go to the NIH ChestX-ray14 dataset on Kaggle:
   - Main dataset: [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
   - Or use the sample version if you want faster experiments.

2. Download the data and place images in the correct folder structure expected by the notebook.

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Run the Notebook

Open `Multi-Label_Chest_XRay_Disease_Classification.ipynb` in Jupyter / VS Code / Colab and run cell by cell.

> **Important**: Update the paths in the `CFG` class at the beginning of the notebook to point to your local dataset location.

---

## 📊 Key Results (from final run)

- **Best Macro AUC**: ~0.758
- **Best Class**: Fibrosis (AUC 0.87)
- Strong performance on most diseases despite heavy imbalance

---

## 🛠️ Supporting Python Files (Recommended)

I recommend you extract the following reusable modules from the notebook:

- `utils/dataset.py` → `ChestXRayDataset` class with Albumentations transforms
- `utils/focal_loss.py` → `FocalLoss` class
- `utils/metrics.py` → Functions for threshold tuning, per-class AUC, F1 etc.
- `utils/utils.py` → `CFG`, `seed_everything`, memory management

These make the code much cleaner and easier for you to reuse.

---

## 📚 Citations

### Dataset

**NIH ChestX-ray14**

> Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2097–2106.

**Official Dataset Page**: https://www.kaggle.com/datasets/nih-chest-xrays/data

### Model Architectures

- **DenseNet-121** (Primary Backbone)

> Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700–4708.

- **Vision Transformer (ViT-B/16)** (used in ablation)

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

### Other References

- **Focal Loss**

> Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

- **Grad-CAM**

> Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

---

## ⚠️ Ethical Considerations

This project is **for research/educational purposes only**. It should **not** be used for clinical diagnosis without proper validation, regulatory approval, and human oversight.

See the "Ethical Considerations" section in the notebook for detailed discussion on bias, fairness, clinical risks, and regulatory compliance.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NIH Clinical Center for releasing the ChestX-ray14 dataset
- Kaggle community for hosting the data and providing GPU resources
- PyTorch, timm, and Albumentations teams

---

## ⭐ Star this repo if you found it helpful!

Feel free to open issues or submit pull requests for improvements.

---

**Made with ❤️ for the medical imaging community**

---
