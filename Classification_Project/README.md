# 🚪 Room Occupancy Detection - Classification Project

## 📌 Project Overview

This project predicts whether a room is **Occupied** or **Empty** using environmental sensor readings.

The dataset contains measurements collected from different sensors inside a room, and the goal is to classify the room occupancy status using Machine Learning classification algorithms.

---

## 🚀 Live Demo

Try the deployed Streamlit application here:

🔗 https://nti-machine-learning-for-data-analysis-gwru4tpcgwnhp7c4r8ia5d.streamlit.app/

---

## 📂 GitHub Repository

🔗 https://github.com/AmrAhmedAttia/NTI-Machine-Learning-for-Data-Analysis

---

## 📂 Dataset Features

Input Features:

- Temperature
- Humidity
- Light
- CO2
- Humidity Ratio

Target:

- Occupancy
  - **0 = Empty**
  - **1 = Occupied**

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

- Data Exploration (EDA)
- Checking Missing Values
- Checking Duplicate Values
- Feature Selection
- Train/Test Split
- Feature Scaling using StandardScaler

---

## 🤖 Machine Learning Models

The following classification models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🏆 Best Model

After comparing all models, the best-performing model was selected and saved.

Saved files:

- `best_classification_model.pkl`
- `scaler.pkl`

---

## 🌐 Streamlit Application

A Streamlit web application was developed to allow users to enter sensor readings and predict whether the room is occupied.

Run the application locally:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Classification_Project/
│
├── app.py
├── Occupancy_Classification.ipynb
├── Occupancy.csv
├── best_classification_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 📦 Required Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- streamlit
- joblib

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/AmrAhmedAttia/NTI-Machine-Learning-for-Data-Analysis.git
```

2. Navigate to the project folder

```bash
cd Classification_Project
```

3. Install the required libraries

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Amr Ahmed Attia**

- Faculty of Computers and Artificial Intelligence
- Menoufia National University
- NTI Machine Learning for Data Analysis Scholarship

---

⭐ If you found this project useful, don't forget to star the repository! occupancy occupancy detection features