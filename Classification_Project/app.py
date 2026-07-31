from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Room Occupancy Detection",
    page_icon="🚪",
    layout="centered"
)

# ==========================
# File Paths
# ==========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_classification_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ==========================
# App Title
# ==========================
st.title("🚪 Room Occupancy Detection")

st.write(
    "Enter the sensor readings below to predict whether the room is occupied."
)

st.markdown("---")

# ==========================
# User Inputs
# ==========================
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=10.0,
        max_value=50.0,
        value=20.9
    )

    light = st.number_input(
        "Light (Lux)",
        min_value=0.0,
        max_value=2000.0,
        value=130.0
    )

    humidity_ratio = st.number_input(
        "Humidity Ratio",
        min_value=0.0,
        max_value=0.01,
        value=0.0042,
        format="%.5f"
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=27.6
    )

    co2 = st.number_input(
        "CO2 (ppm)",
        min_value=300.0,
        max_value=3000.0,
        value=690.0
    )

st.markdown("---")

# ==========================
# Prediction
# ==========================
if st.button("Predict Occupancy 🔍", use_container_width=True):

    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Light": [light],
        "CO2": [co2],
        "HumidityRatio": [humidity_ratio]
    })

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.success("🟢 Room is Occupied")
    else:
        st.error("🔴 Room is Empty")