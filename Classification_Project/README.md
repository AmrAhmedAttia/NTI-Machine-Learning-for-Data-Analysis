# 🚪 Room Occupancy Detection - Classification Project

## 📌 Project Overview

This project predicts whether a room is **Occupied** or **Empty** using environmental sensor readings.

The dataset contains measurements collected from different sensors inside a room, and the goal is to classify the room occupancy status.

---

## 📂 Dataset Features

* Temperature
* Humidity
* Light
* CO2
* Humidity Ratio

**Target:**

* Occupancy

  * 0 = Empty
  * 1 = Occupied

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

* Data Exploration (EDA)
* Checking Missing Values
* Checking Duplicate Values
* Feature Selection
* Train/Test Split
* Feature Scaling using StandardScaler

---

## 🤖 Machine Learning Models

The following classification models were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

## 🏆 Best Model

The best-performing model was saved using Joblib as:

* `best_classification_model.pkl`

The scaler used during preprocessing was also saved as:

* `scaler.pkl`

---

## 🌐 Streamlit Application

A simple Streamlit application was developed to allow users to enter sensor readings and predict whether the room is occupied.

Run the application using:

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
├── occupancy.csv
├── best_classification_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 📦 Required Libraries

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* streamlit
* joblib

Install them using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Amr Ahmed**

Faculty of Computers and Artificial Intelligence

NTI Machine Learning Project
