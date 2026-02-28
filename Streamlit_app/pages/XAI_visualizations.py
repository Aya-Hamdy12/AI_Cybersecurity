# pages/xai.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib
from utils.explainability_xai import explain_single_anomaly
import matplotlib.pyplot as plt

def render_xai():
        # Page Config
    st.set_page_config(page_title="Explainable AI: Anomaly Sample", layout="wide")

    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    # Load datasets and artifacts
    @st.cache_data
    def load_data():
        X_test = joblib.load(os.path.join(BASE_DIR, "Data", "Processed", "test_scaled.pkl"))
        ensemble_df = pd.read_csv(os.path.join(BASE_DIR, "Data", "Processed", "ensemble_preds.csv"))
        
        feature_names_file = os.path.join(BASE_DIR, "Models", "feature_names.pkl")
        if os.path.exists(feature_names_file):
            feature_names = joblib.load(feature_names_file)
        else:
            feature_names = [f"f{i}" for i in range(X_test.shape[1])]
        
        X_test = np.array(X_test)  # Ensure safe indexing
        return ensemble_df, feature_names, X_test

    ensemble_df, feature_names, X_test = load_data()

    st.title("Explainable AI: Anomaly Sample")

    # Sidebar: Select anomaly sample
    detected_anomalies = ensemble_df[ensemble_df['Weighted'] == 1].reset_index()
    sample_index = detected_anomalies.sample(1, random_state=123).index[0]
    st.sidebar.markdown(f"**Total Detected Anomalies:** {len(detected_anomalies)}")
    sample_index = st.sidebar.slider(
        "Select anomaly sample index",
        min_value=0,
        max_value=len(detected_anomalies)-1,
        value=0
    )

    # Get real sample index in X_test
    real_index = detected_anomalies.loc[sample_index, 'index']

    # Sidebar: number of top features
    top_n = st.sidebar.number_input("Top N Features", min_value=3, max_value=20, value=5)

    # Cache SHAP results per sample to avoid recomputation
    @st.cache_data
    def compute_cached_shap(index, top_n=5):
        return explain_single_anomaly(index, top_n=top_n, show_figure=False)

    # Explain selected anomaly
    try:
        top_features, explanation_text, waterfall_fig, model_contrib = compute_cached_shap(real_index, top_n=top_n)
        
        st.subheader("Feature-Level Explanation")
        st.markdown(f"**Ensemble Contribution:** {model_contrib}")
        st.markdown(explanation_text)
        
        st.subheader("Waterfall Plot")
        st.pyplot(waterfall_fig)

        st.subheader("Top Features Data")
        st.dataframe(top_features)
        
    except Exception as e:
        st.error(f"Could not explain sample index {real_index}: {e}")