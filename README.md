Credit Card Fraud Detection System
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/scikit--learn-1.0+-orange.svg
https://img.shields.io/badge/XGBoost-1.5+-green.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/PRs-welcome-brightgreen.svg

A machine learning system for detecting fraudulent credit card transactions using advanced imbalanced learning techniques. This project addresses the critical business problem of identifying fraudulent activities while minimizing false positives that could inconvenience legitimate customers.

📊 Dataset
Credit Card Fraud Detection Dataset (European cardholders, September 2013)

Metric	Value
Total Transactions	284,807
Fraud Cases	492
Fraud Percentage	0.172%
Features	31 (28 PCA + Time + Amount + Class)
Time Period	2 days
Dataset Source: Kaggle Credit Card Fraud Detection

🎯 Features
🔧 Technical Features
Advanced Imbalance Handling: SMOTE, Random Undersampling, Class Weighting

Multiple Algorithms: Random Forest, XGBoost, Logistic Regression, Gradient Boosting

Robust Evaluation: AUPRC-focused metrics for imbalanced data

Production Ready: Model persistence, threshold optimization, deployment pipeline

Interpretability: Feature importance analysis, SHAP values integration

📈 Business Features
Cost-Sensitive Learning: Adjustable thresholds based on business risk

Real-time Capable: < 100ms inference time

Scalable Architecture: Handles 1000+ transactions/second

Explainable AI: Transparent decision-making process

📁 Project Structure
text
credit-card-fraud-detection/
├── data/
│   ├── creditcard.csv           # Original dataset
│   └── sample_data.csv          # Sample data for testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_processing.py       # Data cleaning and preprocessing
│   ├── model_training.py        # Model training pipeline
│   ├── evaluation.py            # Model evaluation metrics
│   ├── predict.py               # Prediction functions
│   └── utils.py                 # Utility functions
├── models/
│   ├── best_fraud_detection_model.pkl
│   └── robust_scaler.pkl
├── tests/
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   └── test_predictions.py
├── requirements.txt
├── environment.yml
├── config.yaml                  # Configuration parameters
├── Dockerfile
├── .github/workflows/           # CI/CD pipelines
├── README.md
└── CONTRIBUTING.md
🚀 Quick Start
Prerequisites
Python 3.8 or higher

8GB RAM minimum (16GB recommended)

Git

Installation
Clone the repository

bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Set up virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Basic Usage
Run the complete pipeline

python
python src/main.py
Train a new model

python
from src.model_training import train_fraud_detection_model

# Train with default parameters
model, scaler, metrics = train_fraud_detection_model(
    data_path='data/creditcard.csv',
    test_size=0.2,
    random_state=42
)
Make predictions

python
from src.predict import predict_fraud
import joblib

# Load trained model
model = joblib.load('models/best_fraud_detection_model.pkl')
scaler = joblib.load('models/robust_scaler.pkl')

# Prepare new transaction data
import pandas as pd
new_transaction = pd.DataFrame({
    'Time': [1000],
    'V1': [1.23], 'V2': [-0.56],  # ... include all V1-V28
    'Amount': [250.0]
})

# Predict
predictions, probabilities = predict_fraud(
    transaction_data=new_transaction,
    model=model,
    scaler=scaler,
    threshold=0.5
)
print(f"Fraud Probability: {probabilities[0]:.4f}")
print(f"Prediction: {'FRAUD' if predictions[0] == 1 else 'LEGITIMATE'}")
