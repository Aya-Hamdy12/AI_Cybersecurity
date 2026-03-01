import streamlit as st
import pandas as pd
import joblib
import os
import time
from datetime import datetime
import plotly.express as px
import json
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(project_root, "..", "feature_names.json")

with open(FEATURES_PATH, "r") as f:
    expected_columns = json.load(f)


# Add the parent folder to sys.path so Python can find explainability_xai.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Streamlit_app.utils.explainability_xai import explain_single_anomaly, columns_names



# Page Config
st.set_page_config(
    page_title="Live Anomaly Detection Stream",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');

* { font-family: 'Rajdhani', sans-serif; color: #E0E0E0; }
body, .main { background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%); }

h1,h2,h3,h4,h5,h6 {
    font-family: 'Orbitron', monospace !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00E5FF;
    text-align: center;
    text-shadow: 0 0 12px rgba(0,229,255,0.45);
}

h1 { font-size: 2.5rem; margin-bottom: 20px; }
h2 { font-size: 2rem; margin-bottom: 25px; }
h3 { font-size: 1.2rem; margin-bottom: 10px; }
p { color: #b0b0b0; }

.metric-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 40px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
    border: 2px solid #00E5FF;
    border-radius: 12px;
    padding: 25px 20px;
    width: 220px;
    text-align: center;
    box-shadow: 0 0 18px rgba(0,229,255,0.18);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.05);
    border-color: #7F5AF0;
    box-shadow: 0 0 28px rgba(127,90,240,0.25);
}

.metric-card h3 { color: #14FFEC; }
.metric-card p { font-size: 0.9rem; }

.stImage img { border-radius: 12px; }

.hologram-wrap {
    width: 100%;
    max-width: 1080px;
    margin: 0 auto 40px auto;
    padding: 28px;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(20,255,236,0.03), rgba(127,90,240,0.02));
    border: 1px solid rgba(20,255,236,0.06);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    display:flex;
    flex-direction: column;
    justify-content:center;
    align-items:center;
}
</style>
""", unsafe_allow_html=True)

# Page Title inside hologram wrapper
st.markdown('<div class="hologram-wrap"><h1>Live Anomaly Detection Stream</h1></div>', unsafe_allow_html=True)


# Load Pre-trained Model
# PROJECT_ROOT = r"C:/Graduation Project/AI_Cybersecurity"
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODEL_PATH = os.path.join(BASE_DIR, "Models", "iso_model.pkl")

try:
    iso_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Isolation Forest model not found. Please add 'iso_model.pkl' to the Models folder.")
    st.stop()

# Session State Initialization
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=["Time", "Trafic", "anomaly"])
if "trend" not in st.session_state:
    st.session_state.trend = pd.DataFrame(columns=["Normal", "Anomaly"])


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
    

# Live Classification Stream
if st.session_state.data is not None and st.button("Start Live Stream"):
    normal_count = 0
    anomaly_count = 0

    # Placeholders
    badge_placeholder = st.empty()
    live_table_placeholder = st.empty()
    trend_placeholder = st.empty()
    histogram_placeholder = st.empty()
    pie_placeholder = st.empty()
    
    # NEW: container for appending attacks
    attacks_container = st.container()

    for i, row in st.session_state.data.iterrows():
        row_df = pd.DataFrame([row])
        pred = iso_model.predict(row_df)[0]

        if pred == 1:
            anomaly_val = 0
            normal_count += 1
            label_text = "Normal"
        else:
            anomaly_val = 1
            anomaly_count += 1
            label_text = "Attack"

        # Current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append to results
        st.session_state.results = pd.concat(
            [st.session_state.results, pd.DataFrame({"Time": [current_time], "Trafic": [label_text], "anomaly": [anomaly_val]})],
            ignore_index=True
        )

        # Show live counters
        badge_placeholder.markdown(
            f"""
            <span style='background-color:#2196F3; color:white; padding:6px 12px; border-radius:8px; margin-right:10px; font-weight:bold;'>Normal: {normal_count}</span>
            <span style='background-color:#f44336; color:white; padding:6px 12px; border-radius:8px; font-weight:bold;'>Attack: {anomaly_count}</span>
            """,
            unsafe_allow_html=True
        )

        # Display live table
        display_df = st.session_state.results[["Time", "Trafic"]].copy()

        def style_label(val):
            if val == "Normal":
                return "background-color: #2196F3; color: white; padding: 2px 4px; border-radius: 5px; font-weight:bold; text-align:center;"
            else:
                return "background-color: #f44336; color: white; padding: 2px 4px; border-radius: 5px; font-weight:bold; text-align:center;"

        live_table_placeholder.dataframe(
            display_df.tail(200).style.applymap(style_label, subset=["Trafic"])
        )

        # =========================
        # APPEND ONLY NEW ATTACKS
        # =========================
        if anomaly_val == 1:
            with attacks_container:
                st.markdown("### Recent Attack")
                st.markdown(f"🔴 **{current_time} — Attack Detected**")

                with st.expander(f"Generate Explanation for {current_time}", expanded=False):

                    x_sample = row.values.reshape(1, -1)

                    with st.spinner("Generating explanation..."):
                        top_features, explanation_text, fig = explain_single_anomaly(x_sample)

                    st.markdown("#### Explanation")
                    st.write(explanation_text)

                    st.markdown("#### SHAP Waterfall")
                    st.pyplot(fig)

                    st.markdown("---")

        # Update trend line chart
        st.session_state.trend = pd.concat(
            [st.session_state.trend, pd.DataFrame({"Normal": [normal_count], "Anomaly": [anomaly_count]})],
            ignore_index=True
        )
        trend_placeholder.line_chart(st.session_state.trend)

        # Update histogram
        fig_hist = px.histogram(
            st.session_state.results,
            x="anomaly",
            color="anomaly",
            color_discrete_map={0: '#2196F3', 1: "#f44336"},
            title="Anomaly Distribution (Live)",
            labels={"anomaly": "Trafic (0=Normal, 1=Attack)"}
        )
        histogram_placeholder.plotly_chart(fig_hist, use_container_width=True)

        # Update pie chart
        counts = st.session_state.results["anomaly"].value_counts().reset_index()
        counts.columns = ["anomaly", "count"]
        fig_pie = px.pie(
            counts,
            names="anomaly",
            values="count",
            color="anomaly",
            color_discrete_map={0: '#2196F3', 1: "#f44336"},
            title="Anomaly Proportion (Live)",
            hole=0.3
        )
        pie_placeholder.plotly_chart(fig_pie, use_container_width=True)

        time.sleep(0.2)
        
# Download Results
if not st.session_state.results.empty:
    csv = st.session_state.results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Live Stream Results",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )