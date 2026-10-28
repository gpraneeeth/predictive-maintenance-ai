# 🏭 Industrial Predictive Maintenance & Machine Health Engine

An end-to-end Machine Learning pipeline designed to predict equipment failures before they happen using industrial sensor telemetry. Built to minimize unplanned downtime and optimize maintenance scheduling in manufacturing environments.

## 🚀 Project Overview
In modern smart manufacturing, unplanned machine downtime costs millions. This project processes multi-sensor factory data, engineers physics-based performance metrics, and trains a cost-weighted XGBoost classifier to flag high-risk equipment failures with high recall.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib
* **Platform:** Google Colab / Local Development
* **Model:** XGBoost Classifier (with class-imbalance handling)

## 📊 Key Features & Engineering
* **Physics-Based Feature Engineering:** Derived thermal gradients (`Temp_Diff`), mechanical power, and tool overstrain index (`Strain = Torque * Tool_wear`) to capture real-world factory physics.
* **Imbalanced Data Handling:** Implemented class-weight adjustments (`scale_pos_weight`) to ensure rare equipment breakdowns are accurately caught.
* **Explainable AI (XAI):** Extracted and visualized feature importances to help plant technicians identify root causes behind potential breakdowns.

## 📈 Model Performance
* Trained on industrial sensor metrics (air/process temperature, rotational speed, torque, tool wear).
* Evaluated using precision, recall, and classification metrics optimized specifically to reduce false negatives (missed failures).

## 🚀 How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/gpraneeeth/predictive-maintenance-ai/edit/main/README.md](https://github.com/gpraneeeth/predictive-maintenance-ai)
