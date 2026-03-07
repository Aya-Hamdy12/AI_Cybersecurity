import streamlit as st
import pandas as pd
import joblib
import os
import time
from datetime import datetime
import plotly.express as px
import json

# Paths
project_root = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(project_root, "..", "feature_names.json")

with open(FEATURES_PATH, "r") as f:
    expected_columns = json.load(f)

# Page Config
st.set_page_config(page_title="Live Anomaly Detection Stream", layout="wide")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');
* { font-family: 'Rajdhani', sans-serif; color: #E0E0E0; }
body, .main { background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%); }
h1,h2,h3,h4,h5,h6 { font-family: 'Orbitron', monospace !important; text-transform: uppercase; letter-spacing: 2px; color: #00E5FF; text-align: center; text-shadow: 0 0 12px rgba(0,229,255,0.45); }
h1 { font-size: 2.5rem; margin-bottom: 20px; }
h2 { font-size: 2rem; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>Live Anomaly Detection Stream</h1>', unsafe_allow_html=True)

# Load Model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "iso_model.pkl")

try:
    iso_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Isolation Forest model not found. Please add 'iso_model.pkl' to the Models folder.")
    st.stop()

# Initialize session_state variables
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=["Time", "Trafic", "anomaly"])
if 'trend' not in st.session_state:
    st.session_state.trend = pd.DataFrame(columns=["Normal", "Anomaly"])
if 'attack_samples' not in st.session_state:
    st.session_state.attack_samples = []
if 'last_anomaly' not in st.session_state:
    st.session_state.last_anomaly = None

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file for anomaly detection", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if len(df.columns) == len(expected_columns):
        df.columns = expected_columns
        st.success("Column names aligned with training features.")
    else:
        st.error("Feature count mismatch with trained model.")
        st.stop()
    st.session_state.data = df
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
# Start Live Stream
if st.session_state.get('data') is not None and st.button("Start Live Stream"):
    normal_count = 0
    anomaly_count = 0

    badge_placeholder = st.empty()
    live_table_placeholder = st.empty()
    trend_placeholder = st.empty()
    histogram_placeholder = st.empty()
    pie_placeholder = st.empty()

    for i, row in st.session_state.data.iterrows():  # <-- i هو رقم السطر
        row_df = pd.DataFrame([row])
        pred = iso_model.predict(row_df)[0]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x_sample = row.values.reshape(1, -1)

        if pred == 1:
            anomaly_val = 0
            normal_count += 1
            label_text = "Normal"
        else:
            anomaly_val = 1
            anomaly_count += 1
            label_text = "Attack"

            # Save anomaly for XAI with original index
            st.session_state.attack_samples.append({
                "time": current_time,
                "data": x_sample,
                "index": i  
            })
            st.session_state.last_anomaly_index = i

        # Append results
        st.session_state.results = pd.concat([
            st.session_state.results,
            pd.DataFrame({"Time": [current_time], "Trafic": [label_text], "anomaly": [anomaly_val]})
        ], ignore_index=True)

        # Update counters
        badge_placeholder.markdown(
            f"""
            <span style='background-color:#2196F3; color:white; padding:6px 12px; border-radius:8px; margin-right:10px; font-weight:bold;'>Normal: {normal_count}</span>
            <span style='background-color:#f44336; color:white; padding:6px 12px; border-radius:8px; font-weight:bold;'>Attack: {anomaly_count}</span>
            """, unsafe_allow_html=True)

        # Update live table
        def style_label(val):
            return "background-color:#2196F3; color:white; font-weight:bold" if val == "Normal" else "background-color:#f44336; color:white; font-weight:bold"

        live_table_placeholder.dataframe(st.session_state.results.tail(200).style.applymap(style_label, subset=["Trafic"]))

        # Update trend
        st.session_state.trend = pd.concat([
            st.session_state.trend,
            pd.DataFrame({"Normal": [normal_count], "Anomaly": [anomaly_count]})
        ], ignore_index=True)
        trend_placeholder.line_chart(st.session_state.trend)

        # Histogram
        fig_hist = px.histogram(st.session_state.results, x="anomaly", color="anomaly",
                                color_discrete_map={0: "#2196F3", 1: "#f44336"}, title="Anomaly Distribution (Live)")
        histogram_placeholder.plotly_chart(fig_hist, use_container_width=True)

        # Pie chart
        counts = st.session_state.results['anomaly'].value_counts().reset_index()
        counts.columns = ['anomaly', 'count']
        fig_pie = px.pie(counts, names='anomaly', values='count', color='anomaly',
                         color_discrete_map={0: "#2196F3", 1: "#f44336"}, title="Anomaly Proportion (Live)", hole=0.3)
        pie_placeholder.plotly_chart(fig_pie, use_container_width=True)

        time.sleep(0.2)