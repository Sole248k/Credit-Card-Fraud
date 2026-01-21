Credit Card Fraud Detection System 🔒
📌 Project Overview
A production-ready machine learning system that detects fraudulent credit card transactions with 88.59% Average Precision (AUPRC). Designed to handle extreme class imbalance (only 0.172% fraud cases), this system balances fraud detection with customer experience by optimizing the precision-recall trade-off.

🏆 Key Results
Metric	Value	Business Impact
Best Model	Gradient Boosting with SMOTE	Most accurate fraud detection
AUPRC	0.8859	Primary success metric for imbalanced data
Recall	0.85	Catches 85% of fraudulent transactions
Precision	0.79	79% of alerts are genuine fraud cases
F1-Score	0.8193	Balanced performance measure
🏗️ Project Architecture
1. Data Pipeline
python
# Key preprocessing steps:
- RobustScaler() for 'Time' and 'Amount' (handles outliers)
- PCA features (V1-V28) preserved as-is
- Stratified train-test split (80-20)
- Three imbalance handling strategies tested
2. Model Performance Comparison
Model	                Sampling	AUPRC	   F1-Score	    Recall	    Precision
Gradient Boosting	    SMOTE	    0.8859	   0.8193	    0.850	    0.790
XGBoost	                SMOTE	    0.8493	   0.8250	    0.825	    0.825
Random Forest	        SMOTE	    0.8392	   0.8451	    0.750	    0.964
Logistic Regression	    None	    0.5266	   0.2042	    0.850	    0.117
📊 Performance Analysis
Confusion Matrix (Threshold = 0.5)

                    Predicted
                  Non-Fraud  Fraud
Actual  Non-Fraud  16,607     9
        Fraud          6      34
Key Statistics:

True Positives: 34 (correctly identified fraud)

False Positives: 9 (legitimate transactions flagged)

False Negatives: 6 (missed fraud cases)

True Negatives: 16,607 (correctly approved transactions)

Threshold Optimization
The system supports dynamic threshold adjustment for different business needs:

Threshold	Precision	Recall	Business Scenario
0.1	0.163	0.925	High Security - Catch most fraud, accept more false positives
0.3	0.528	0.875	Balanced approach
0.5	0.790	0.850	Default Setting
0.7	0.906	0.775	Conservative approach
0.9	0.971	0.600	Customer Experience Focus - Minimize false positives
