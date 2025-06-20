# insurance-fraud-detection
Fraud detection project using ML models
# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('data/insurance_claims.csv')

# Display basic info
print("Initial data shape:", df.shape)
print(df.head())

# Drop irrelevant columns
df.drop(columns=['policy_number', 'policy_bind_date', 'incident_location', 'incident_date', 'auto_make', 'auto_model'], inplace=True)

# Handle categorical features
df = pd.get_dummies(df, drop_first=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features and target
X = df.drop('fraud_reported_Y', axis=1)
y = df['fraud_reported_Y']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("After SMOTE:", X_res.shape)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Save models (optional)
import joblib
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(xgb, 'xgboost_model.pkl')
📄 README.md (Project Summary)
markdown
Copy
Edit
# 🛡️ Insurance Fraud Detection Using Machine Learning

This project identifies fraudulent insurance claims using machine learning models like Random Forest and XGBoost. The dataset is cleaned, engineered, and modeled to improve fraud prediction accuracy.

## 🚀 Features
- Data preprocessing
- SMOTE for class imbalance
- Random Forest & XGBoost models
- Performance evaluation
- Model serialization

