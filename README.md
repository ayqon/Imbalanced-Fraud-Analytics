Module: WM9B7 - Artificial Intelligence and Deep Learning
**Student:** Ioannis Konstantinou | **Student ID:** u5750302<br>
**Assessment:** Individual Assessment - Credit Card Fraud Detection<br>
**Module Leader:** Dr Leonardo Alves Dias<br>
**Submission Date:** 27 April 2025

# Credit Card Fraud Detection: A Deep Learning Approach

---

## README and Setup Guide

### Problem Description
Credit card fraud is a significant financial crime that causes billions of dollars in
losses each year. This notebook develops and compares multiple machine learning and
deep learning approaches to detect fraudulent transactions from anonymised credit card
data. The core challenge is severe class imbalance, where fraud cases represent only
approximately 0.17% of all transactions.

### Dataset Description
**Source:** [Kaggle - Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**File:** creditcard.csv
- 284,807 transactions with 492 fraud cases (0.17%)
- Features V1 to V28: PCA-transformed anonymised features
- Time: seconds elapsed since the first transaction
- Amount: transaction amount in EUR
- Class: target variable (1 = Fraud, 0 = Legitimate)
- Licence: CC0 Public Domain

---

### How to Run
1. Open this notebook in VS Code or Jupyter
2. Press Run All
3. The setup cell installs all required libraries automatically and downloads
   the dataset directly from Kaggle without requiring an account or token

No manual configuration is needed. The dataset download is handled programmatically
on every run and cached locally after the first execution.

---

### Required Libraries
All libraries are installed automatically with pinned versions by the first cell:

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical operations |
| matplotlib | 3.9.0 | Visualisation |
| seaborn | 0.13.2 | Statistical plots |
| scikit-learn | 1.5.0 | Traditional ML models, preprocessing, evaluation |
| tensorflow | 2.16.1 | Deep learning models (ANN, LSTM, Autoencoder) |
| xgboost | 2.0.3 | Gradient boosting |
| imbalanced-learn | 0.12.3 | SMOTE oversampling |
| kagglehub | 0.3.4 | Automatic dataset download |

---

### Academic Note
This notebook critically compares traditional machine learning, deep learning, and
unsupervised anomaly detection methods. Deep learning is not assumed to outperform
classical methods, particularly on tabular data. Design decisions, trade-offs, and
limitations are discussed throughout.
