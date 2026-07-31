# 🏠 California Housing Price Prediction

A Machine Learning Regression project that predicts **California median house prices** using multiple regression algorithms and provides an interactive web application built with **Streamlit**.

## 🚀 Live Demo

🔗 https://nti-machine-learning-for-data-analysis-bvhcyf4iwysqu6sn6xjf6y.streamlit.app/

---

## 📌 Project Overview

This project compares multiple regression algorithms to predict housing prices using the California Housing dataset.

The workflow includes:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Interactive prediction with Streamlit

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

🏆 **Best Model:** XGBoost

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| Best Model | XGBoost |
| R² Score | **0.8339** |
| RMSE | **46,647** |
| MAE | **30,894** |

---

## ✨ Streamlit Features

- 📂 Upload your own housing dataset
- 📈 Compare multiple regression models
- 📊 Display model evaluation metrics
- 🔍 Explore the dataset
- 🏠 Predict California house prices instantly
- 📉 Visualize Feature Importance

---

## 📁 Project Structure

```text
regression_project/
│
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── housing.csv
│
├── models/
│   ├── best_regression_model.pkl
│   └── scaler_regression.pkl
│
└── notebooks/
    └── housing_regression_analysis.ipynb
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Matplotlib

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/AmrAhmedAttia/NTI-Machine-Learning-for-Data-Analysis.git
```

Go to the project

```bash
cd regression_project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📂 Dataset

The project uses the **California Housing Dataset**.

Required columns:

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity
- median_house_value

---

## 📸 Application Preview

After deployment, the application provides:

- Interactive prediction interface
- Model comparison dashboard
- Dataset overview
- Feature importance visualization

---

## 👨‍💻 Author

**Amr Ahmed Attia**

Computer Science & Artificial Intelligence Student

- GitHub: https://github.com/AmrAhmedAttia

---

## ⭐ If you like this project

Give the repository a ⭐ on GitHub.