from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET = "median_house_value"
EXPECTED_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    TARGET,
]
NUMERIC_FEATURES = [column for column in EXPECTED_COLUMNS if column not in {"ocean_proximity", TARGET}]
CATEGORICAL_FEATURES = ["ocean_proximity"]


def load_dataset(uploaded_file: BinaryIO | None = None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        dataset_path = Path(__file__).resolve().parent / "data" / "housing.csv"

        if not dataset_path.exists():
            raise FileNotFoundError("housing.csv was not found")

        df = pd.read_csv(dataset_path)

    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"The dataset is missing these required columns: {', '.join(missing)}")

    return df[EXPECTED_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


@st.cache_resource(show_spinner=False)
def _train_cached(dataset_csv: str) -> dict:
    from io import StringIO

    df = pd.read_csv(StringIO(dataset_csv))
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, predictions),
            }
        )
        trained_models[name] = pipeline

    comparison = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_model_name = comparison.iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]

    importance = None
    estimator = best_pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        clean_names = [name.split("__", 1)[-1] for name in feature_names]
        importance = pd.Series(
            estimator.feature_importances_, index=clean_names, name="Importance"
        ).sort_values(ascending=False)

    return {
        "models": trained_models,
        "comparison": comparison,
        "best_model_name": best_model_name,
        "best_model": best_pipeline,
        "feature_importance": importance,
    }


def train_models(df: pd.DataFrame) -> dict:
    return _train_cached(df.to_csv(index=False))


def make_prediction(bundle: dict, input_data: dict) -> float:
    sample = pd.DataFrame([input_data])
    prediction = bundle["best_model"].predict(sample)[0]
    return max(0.0, float(prediction))
