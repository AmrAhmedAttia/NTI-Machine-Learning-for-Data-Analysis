from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import (
    EXPECTED_COLUMNS,
    load_dataset,
    train_models,
    make_prediction,
)

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 California Housing Price Predictor")
st.caption("Regression project comparing Linear Regression, Decision Tree, Random Forest, and XGBoost.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload housing.csv", type="csv")
    st.info("You can also place `housing.csv` in the project root.")

try:
    df = load_dataset(uploaded_file)
except FileNotFoundError:
    st.warning("Upload `housing.csv` from the sidebar, or add it to the project folder to start the app.")
    st.markdown(
        "The dataset must contain the California Housing columns: "
        "`longitude`, `latitude`, `housing_median_age`, `total_rooms`, "
        "`total_bedrooms`, `population`, `households`, `median_income`, "
        "`ocean_proximity`, and `median_house_value`."
    )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.spinner("Training and evaluating regression models..."):
    bundle = train_models(df)

comparison = bundle["comparison"]
best_model_name = bundle["best_model_name"]

metric1, metric2, metric3, metric4 = st.columns(4)
best_row = comparison.iloc[0]
metric1.metric("Best model", best_model_name)
metric2.metric("R²", f"{best_row['R2']:.4f}")
metric3.metric("RMSE", f"${best_row['RMSE']:,.0f}")
metric4.metric("MAE", f"${best_row['MAE']:,.0f}")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "🔍 Data Overview"])

with tab1:
    st.subheader("Enter district information")

    c1, c2, c3 = st.columns(3)
    with c1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
        latitude = st.number_input("Latitude", value=37.88, format="%.4f")
        housing_median_age = st.number_input("Housing median age", min_value=1.0, value=41.0)
    with c2:
        total_rooms = st.number_input("Total rooms", min_value=1.0, value=880.0)
        total_bedrooms = st.number_input("Total bedrooms", min_value=1.0, value=129.0)
        population = st.number_input("Population", min_value=1.0, value=322.0)
    with c3:
        households = st.number_input("Households", min_value=1.0, value=126.0)
        median_income = st.number_input("Median income", min_value=0.0, value=8.3252, format="%.4f")
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            sorted(df["ocean_proximity"].dropna().astype(str).unique()),
        )

    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    if st.button("Predict house value", type="primary", use_container_width=True):
        prediction = make_prediction(bundle, input_data)
        st.success(f"Estimated median house value: **${prediction:,.2f}**")
        st.caption(f"Prediction generated with the best evaluated model: {best_model_name}.")

with tab2:
    st.subheader("Model comparison")
    display_df = comparison.copy()
    for col in ["MAE", "MSE", "RMSE"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}")
    display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("R² score by model")
    st.bar_chart(comparison.set_index("Model")[["R2"]])

    st.subheader(f"Feature importance — {best_model_name}")
    importance = bundle["feature_importance"]
    if importance is not None:
        st.bar_chart(importance.head(12))
    else:
        st.info("Feature importance is not available for this model.")

with tab3:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("Target distribution")
    st.bar_chart(df["median_house_value"].value_counts(bins=20).sort_index())

st.divider()
st.caption("Built with Streamlit, scikit-learn, and XGBoost.")
