import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="centered")
st.title("🌍 Air Quality Index (AQI) Prediction")
st.write("Enter pollutant concentrations to estimate AQI and classify the pollution level.")

reg_path = MODEL_DIR / "aqi_regressor.joblib"
clf_path = MODEL_DIR / "aqi_classifier.joblib"
if not reg_path.exists() or not clf_path.exists():
    st.error("Trained models are missing. Run: python train_model.py")
    st.stop()

regressor = joblib.load(reg_path)
classifier = joblib.load(clf_path)

st.subheader("Pollutant values")
col1, col2 = st.columns(2)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=35.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1500.0, value=60.0)
    no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, max_value=1000.0, value=25.0)
with col2:
    so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, max_value=1000.0, value=10.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=100.0, value=1.0)
    o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, max_value=1000.0, value=30.0)

if st.button("Predict AQI", type="primary", use_container_width=True):
    values = pd.DataFrame([[pm25, pm10, no2, so2, co, o3]],
        columns=["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"])
    predicted_aqi = max(0, min(500, int(round(regressor.predict(values)[0]))))
    predicted_level = classifier.predict(values)[0]
    st.metric("Predicted AQI", predicted_aqi)
    st.success(f"Predicted pollution level: **{predicted_level}**")
    guidance = {
        "Good": "Air quality is considered satisfactory.",
        "Satisfactory": "Generally acceptable; unusually sensitive people may experience minor effects.",
        "Moderate": "Sensitive people should consider reducing prolonged outdoor exertion.",
        "Poor": "Reduce prolonged outdoor exertion, especially for sensitive groups.",
        "Very Poor": "Avoid prolonged outdoor activity; follow local health advisories.",
        "Severe": "Avoid outdoor exertion and follow official public-health guidance."
    }
    st.info(guidance.get(predicted_level, "Follow local air-quality guidance."))
