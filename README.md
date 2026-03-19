# Stacking Ensemble XAI

Stacking ensemble model with SHAP-based explainability using Scikit-learn and XGBoost.

---

## 📌 Overview

This project implements a **stacking ensemble learning model** that combines multiple base classifiers to improve prediction performance. It also integrates **SHAP (SHapley Additive exPlanations)** to provide interpretability and insights into model predictions.

---

## 🚀 Features

- 🔹 Stacking ensemble with multiple base models
- 🔹 Models used: Random Forest, SVM, XGBoost
- 🔹 Meta-learner: Logistic Regression
- 🔹 Cross-validation for robust training
- 🔹 SHAP-based explainability for model interpretation
- 🔹 Improved performance over individual models

---

## 🧠 Model Architecture

**Base Models:**

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

**Meta Model:**

- Logistic Regression

The predictions from base models are used as inputs to the meta-model for final prediction.

---

## 📊 Explainability with SHAP

- Visualizes feature importance
- Explains individual predictions
- Helps understand model behavior

Example outputs:

- SHAP summary plots
- Feature importance graphs
- Force plots

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- XGBoost
- SHAP
- NumPy / Pandas
- Matplotlib / Seaborn

---

## 📈 Results

- Achieved improved accuracy compared to individual base models
- Robust performance using cross-validation
- Interpretable predictions using SHAP

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV / Optuna
- Add more base models
- Deploy as API (FastAPI)
- Dashboard for SHAP visualizations

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
