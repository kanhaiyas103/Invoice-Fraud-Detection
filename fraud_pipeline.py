# 🚀 fraud_pipeline.py - Starter Code for Vendor Invoice Fraud Detection

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import datetime

# === 1. Load the Data ===
data = pd.read_csv('data/synthetic_invoices_test.csv')

# === 2. Preprocessing ===
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time

# Drop duplicates
original_len = len(data)
data.drop_duplicates(inplace=True)
print(f"Removed {original_len - len(data)} duplicate rows.")

# === 3. Manual Rule-Based Checks ===
data['is_duplicate_invoice'] = data.duplicated(['Invoice Number'], keep=False).astype(int)
data['is_amount_outlier'] = ((data['Amount'] > data['Amount'].mean() + 2 * data['Amount'].std()) |
                              (data['Amount'] < data['Amount'].mean() - 2 * data['Amount'].std())).astype(int)
data['is_weird_time'] = data['Time'].apply(lambda t: 1 if t < datetime.time(6,0) or t > datetime.time(22,0) else 0)

# === 4. Prepare Data for ML Models ===
ml_features = ['Amount']
X = data[ml_features]

# === 5. Apply ML Anomaly Detection Models ===
iso_forest = IsolationForest(contamination=0.05, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
oneclass = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)

# Fit/Transform Models 
data['IF_outlier'] = iso_forest.fit_predict(X)
data['LOF_outlier'] = lof.fit_predict(X)
data['SVM_outlier'] = oneclass.fit_predict(X) 

# Convert model outputs to 1 = anomaly, 0 = normal
for col in ['IF_outlier', 'LOF_outlier', 'SVM_outlier']:
    data[col] = data[col].apply(lambda x: 1 if x == -1 else 0)

# === 6. Combine Results ===
data['fraud_risk_score'] = data[['is_duplicate_invoice', 'is_amount_outlier', 'is_weird_time',
                                 'IF_outlier', 'LOF_outlier', 'SVM_outlier']].sum(axis=1)
data['fraud_label'] = data['fraud_risk_score'].apply(lambda x: 1 if x >= 3 else 0)

# === 7. Save Output ===
data.to_csv('output/invoices_with_risk_score.csv', index=False)
print("✅ Fraud risk scores saved to output/invoices_with_risk_score.csv") 