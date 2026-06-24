# graduation-project
# 🌱 Precision Agriculture: Crop Water Prediction API

## Overview
An end-to-end machine learning pipeline and REST API designed for precision agriculture. This system acts as the AI backend for a hardware system integrating an NPK 8-in-1 soil sensor. It predicts the exact daily water requirement (in mm/day) for various crops based on real-time environmental telemetry, soil conditions, and biological growth stages. 

## 🚀 Features
* **High-Precision ML Engine:** Utilizes a highly regularized `HistGradientBoostingRegressor` to capture complex, non-linear relationships between weather, soil, and crop biology.
* **Zero Data Leakage Pipeline:** Robust preprocessing using `ColumnTransformer` (StandardScaler, OneHotEncoder, OrdinalEncoder) built to cleanly separate train and test environments.
* **Cross-Validated Stability:** Model performance mathematically proven via 5-Fold Cross-Validation, ensuring zero overfitting with a highly stable Mean Absolute Error (MAE) of ~0.33 mm/day.
* **Lightning Fast Web API:** Deployed using FastAPI and Uvicorn for asynchronous, high-throughput IoT sensor integrations.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Backend API:** FastAPI, Uvicorn, Pydantic
* **Visualization & EDA:** Matplotlib, Seaborn, Plotly

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/graduation-project.git](https://github.com/YourUsername/graduation-project.git)
   cd graduation-project
