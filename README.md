# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📌 Project Overview
This project implements an **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions. Given the **extreme class imbalance** (only **0.172% fraudulent transactions**), traditional accuracy metrics are misleading. Therefore, the system prioritizes **Average Precision (AUPRC)** to ensure strong fraud detection while minimizing disruption to legitimate users.

This project is designed with **real-world financial applications** in mind and demonstrates best practices in handling imbalanced datasets.

---

## 🎯 Business Problem
Financial institutions lose billions annually due to undetected fraud, while overly aggressive systems frustrate legitimate customers.  
**Goal:** Detect fraudulent transactions effectively **without blocking the 99.8% of legitimate users**.

---

## 📊 Key Results

- **Best Model:** Gradient Boosting Classifier  
- **Sampling Strategy:** SMOTE (10% minority class representation)  
- **Average Precision (AUPRC):** **0.8859**  
- **Recall:** **0.85** (85% of fraud detected)  
- **Precision:** **0.79** (Low false-alarm rate)

---

## 🏗 Project Architecture

### 1. Data Pipeline

**Feature Scaling**
- `Time` and `Amount` scaled using **RobustScaler** to reduce outlier impact
- PCA features (`V1–V28`) preserved as provided

**Class Imbalance Handling**
- Baseline (No Sampling)
- SMOTE Oversampling
- Random Undersampling

---

### 2. Model Evaluation Summary

| Model                | Sampling | AUPRC  | F1-Score | Recall |
|---------------------|----------|--------|----------|--------|
| Gradient Boosting   | SMOTE    | 0.8859 | 0.8193   | 0.850  |
| XGBoost             | SMOTE    | 0.8493 | 0.8250   | 0.825  |
| Random Forest       | SMOTE    | 0.8392 | 0.8451   | 0.750  |
| Logistic Regression | None     | 0.5266 | 0.2042   | 0.850  |

---

## 📈 Performance Analysis

### Precision–Recall Trade-off
The classification threshold can be adjusted based on business priorities:

- **High Security Mode (Threshold = 0.1)**  
  - Recall: **0.925**  
  - Maximizes fraud detection at the cost of higher false positives

- **Low Customer Friction Mode (Threshold = 0.9)**  
  - Precision: **0.971**  
  - Minimizes customer disruption

---

### Confusion Matrix (Final Model)

Test Set Size: **16,656 transactions**

- **True Positives:** 34  
- **False Positives:** 9  
- **False Negatives:** 6  
- **True Negatives:** 16,607  

---

## 🔍 Feature Importance

The Gradient Boosting model provides insight into which features most strongly influence fraud detection.

**Key Observations:**
- PCA components such as `V14`, `V10`, and `V12` consistently rank among the most predictive features
- Transaction `Amount` has moderate influence, indicating higher-value transactions carry increased fraud risk
- `Time` contributes minimally, suggesting fraud is not time-dependent in this dataset

> Feature importance analysis improves **model interpretability**, a critical requirement for financial and regulatory environments.

---

## 🛠 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

### Run the Model
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

sm = SMOTE(sampling_strategy=0.1, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model = GradientBoostingClassifier()
model.fit(X_res, y_res)
```

---

## 🧠 Skills Demonstrated
- Imbalanced classification techniques (SMOTE, AUPRC optimization)
- Model evaluation beyond accuracy
- Business-driven threshold tuning
- Interpretable machine learning for finance
- Production-oriented ML pipeline design

---

## 📜 Conclusion
Combining **SMOTE** with **Gradient Boosting** delivers the best balance between fraud detection and customer experience. This approach is well-suited for deployment in real-world financial systems where minimizing false positives is just as important as catching fraud.

---

## 📄 License
This project is licensed under the **MIT License**.
