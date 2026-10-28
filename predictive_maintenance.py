"""
Project: Predictive Maintenance & Machine Health Engine
Description: End-to-end machine learning pipeline analyzing industrial 
sensor telemetry to predict equipment failure and minimize downtime.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb

def main():
    print("1. Loading dataset...")
    url = "https://raw.githubusercontent.com/SamyamoyRakshit/AI4I-2020-Predictive-Maintenance-Dataset__Linear-Regression/main/ai4i2020.csv"
    df = pd.read_csv(url)

    print("2. Engineering physics-based features...")
    # Drop unnecessary identification columns
    df_cleaned = df.drop(columns=['UDI', 'Product ID'])

    # Engineer thermal and mechanical features
    df_cleaned['Temp_Diff'] = df_cleaned['Process temperature [K]'] - df_cleaned['Air temperature [K]']
    
    omega = df_cleaned['Rotational speed [rpm]']
    df_cleaned['Power'] = df_cleaned['Torque [Nm]'] * (omega * 2 * np.pi / 60)
    df_cleaned['Strain'] = df_cleaned['Tool wear [min]'] * df_cleaned['Torque [Nm]']

    # One-hot encode categorical features ('Type')
    df_cleaned = pd.get_dummies(df_cleaned, columns=['Type'], drop_first=True)

    # Drop specific sub-failure flags to prevent data leakage
    sub_failures = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    existing_sub_failures = [col for col in sub_failures if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=existing_sub_failures)

    # Clean column names for XGBoost compatibility (remove brackets/special chars)
    df_cleaned = df_cleaned.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))

    print("3. Preparing data for training...")
    X = df_cleaned.drop(columns=['Machine_failure'])
    y = df_cleaned['Machine_failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class weight to handle rare failure events
    failure_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    print("4. Training XGBoost Classifier...")
    model = XGBClassifier(
        scale_pos_weight=failure_ratio,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    print("5. Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("6. Plotting Feature Importances...")
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, height=0.6, color='dodgerblue')
    plt.title("Top Sensor & Engineering Features Driving Machine Failures", fontsize=12, fontweight='bold')
    plt.xlabel("F-Score (Importance)", fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()
