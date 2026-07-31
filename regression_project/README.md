# California Housing Regression Project 🏠

A machine learning regression project that predicts California median house values and compares four models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The repository includes an exploratory notebook and an interactive Streamlit application for model comparison, feature-importance visualization, and live predictions.

## Project Results

Results recorded in the original notebook:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 31,887.39 | 48,266.27 | **0.8222** |
| Random Forest | **31,631.02** | 49,008.79 | 0.8167 |
| Decision Tree | 44,120.68 | 69,680.88 | 0.6295 |
| Linear Regression | 50,670.74 | 70,060.52 | 0.6254 |

XGBoost achieved the highest R² score in the notebook experiment.

## Repository Structure

```text
regression_project/
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── housing.csv
├── models/
│   ├── best_regression_model.pkl
│   └── scaler_regression.pkl
└── notebooks/
    └── housing_regression_analysis.ipynb
```

## Dataset

The app expects the California Housing CSV commonly available as `housing.csv` with these columns:

```text
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, ocean_proximity, median_house_value
```

You can either:

1. Put `housing.csv` in the project root, or
2. Upload it from the Streamlit sidebar after starting the app.

## Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/regression_project.git
cd regression_project

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Streamlit
streamlit run app.py
```

Then open the local URL shown in your terminal, usually `http://localhost:8501`.

## Streamlit Features

- Upload the housing dataset directly from the sidebar
- Automatically train and compare four regression models
- Display MAE, MSE, RMSE, and R²
- Select the best model by R² score
- Show feature importance when supported
- Predict a median house value from user inputs

## Methodology

1. Validate the dataset columns.
2. Split the data into training and testing sets using an 80/20 split.
3. Impute missing numeric values with the median.
4. Standardize numeric features.
5. One-hot encode `ocean_proximity`.
6. Train and evaluate four regression algorithms.
7. Select the model with the highest test R² score.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and select your repository.
4. Set the main file path to `app.py`.
5. Deploy the app.

Because the dataset may not be committed, users can upload `housing.csv` directly through the deployed application.

## Author

**Amr Ahmed Attia**

Computer Science & Artificial Intelligence Student

## License

This project is intended for educational and portfolio use.
