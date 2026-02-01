# An Explainable Deep Neural Network Approach for Non-Hodgkin Lymphoma Classification

## Overview
This project presents an explainable deep learning framework for the automated classification of Non-Hodgkin Lymphoma (NHL) subtypes:

- Mantle Cell Lymphoma (MCL)  
- Follicular Lymphoma (FL)  
- Chronic Lymphocytic Leukemia (CLL)

The framework integrates deep neural networks with Explainable Artificial Intelligence (XAI) techniques such as Grad-CAM and SHAP to ensure accurate and interpretable predictions for clinical decision support.

---

## Objectives
- Develop a deep neural network for multi-class NHL subtype classification (MCL, FL, CLL)
- Preprocess and standardize histopathological image datasets
- Evaluate performance using accuracy, precision, recall, and F1-score
- Integrate Grad-CAM for visual interpretation
- Integrate SHAP for feature-level explanation
- Design a two-stage framework combining classification and explainability
- Enhance model reliability and clinical trustworthiness

---

## Methodology
1. Data Preprocessing  
   - Image resizing and normalization  
   - Data augmentation  
   - Dataset split: Training (70%), Validation (15%), Testing (15%)

2. Model Architecture  
   - Hybrid Vision Transformer (H-ViT)  
   - CNN backbone (MobileNetV2) for local feature extraction  
   - Vision Transformer for global contextual representation  
   - Fully connected layers with Softmax classifier  

3. Explainability  
   - Grad-CAM: Visual localization of important regions  
   - SHAP: Feature contribution analysis  

4. Evaluation Metrics  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix  

---

## System Architecture
Pipeline:
1. Input histopathological images  
2. Preprocessing and augmentation  
3. Feature extraction using Hybrid CNN–Transformer model  
4. Classification into MCL, FL, or CLL  
5. Explainability using Grad-CAM and SHAP  
6. Performance evaluation and visualization  

---

## Technical Specifications

### Software Requirements
- Python 3.10+
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab

### Hardware Requirements
- Processor: Intel i5 / AMD Ryzen 5 or higher  
- RAM: Minimum 8 GB (Recommended 16 GB)  
- GPU: NVIDIA GPU (T4 / RTX preferred)  
- Storage: Minimum 50 GB  

---

## Features
- Multi-class NHL subtype classification  
- High accuracy with deep learning  
- Visual explanations using Grad-CAM  
- Feature-level explanations using SHAP  
- Modular and scalable architecture  
- Research-oriented and reproducible pipeline  

---


## Results
The framework demonstrates effective classification of NHL subtypes with strong performance metrics while providing meaningful visual and feature-based explanations, supporting transparent and reliable clinical decision-making.

---

## Explainable AI (XAI)
- Grad-CAM identifies regions of interest influencing predictions  
- SHAP provides numerical attribution values for features  
- Ensures interpretability, accountability, and trustworthiness  

---

## Scope and Limitations
- Limited to MCL, FL, and CLL subtypes  
- Dataset size and quality affect performance  
- Not intended for real-time clinical deployment  
- Designed as a decision-support system, not a replacement for pathologists  

---

## Authors
- Viren Deepak Jawadwar (22BCE3411)  
- Sahil N (22BCE3762)  
- H. Preyansh Nahar (22BCE3807)  

Under the supervision of  
Dr. Diviya M  
Assistant Professor, SCOPE  
VIT University  

---

## License
This project is intended for academic and research purposes only.

---

## Contact
GitHub: https://github.com/preyanshnahar

---

## Acknowledgment
This project was developed as part of the B.Tech Capstone Project in Computer Science and Engineering, focusing on Explainable Artificial Intelligence in medical diagnosis.
