import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Room Occupancy Detection", page_icon="🚪")


@st.cache_resource
def load_model():
    model = joblib.load("best_classification_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error("❌ Failed to load model")
    st.exception(e)

    from pathlib import Path
    import os

    BASE_DIR = Path(__file__).parent

    st.write("Current working directory:")
    st.code(os.getcwd())

    st.write("Files inside project folder:")
    st.write(os.listdir(BASE_DIR))

    st.stop()


st.title("🚪 Room Occupancy Detection")
st.write("أدخل قراءات المستشعرات أدناه لمعرفة ما إذا كانت الغرفة مشغولة أم فارغة.")

st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C) - درجة الحرارة", min_value=10.0, max_value=50.0, value=20.9)
    light = st.number_input("Light (Lux) - الإضاءة", min_value=0.0, max_value=2000.0, value=130.0)
    humidity_ratio = st.number_input("Humidity Ratio - نسبة الرطوبة", min_value=0.0, max_value=0.01, value=0.0042, format="%.5f")

with col2:
    humidity = st.number_input("Humidity (%) - الرطوبة", min_value=0.0, max_value=100.0, value=27.6)
    co2 = st.number_input("CO2 (ppm) - ثاني أكسيد الكربون", min_value=300.0, max_value=3000.0, value=690.0)

st.markdown("---")

if st.button("Predict Occupancy 🔍", use_container_width=True):
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Light': [light],
        'CO2': [co2],
        'HumidityRatio': [humidity_ratio]
    })
    
    scaled_data = scaler.transform(input_data)
    
  
    prediction = model.predict(scaled_data)[0]
    
  
    if prediction == 1:
        st.success("🟢 **النتيجة:** الغرفة مشغولة (Occupied)")
    else:
        st.info("🔴 **النتيجة:** الغرفة فارغة (Empty)")