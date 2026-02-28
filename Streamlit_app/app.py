# app.py
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Anomaly Detection", "Visualizations", "Explainability (XAI)"])

# Home Page
if page == "Home":
    st.markdown("<h1 style='text-align:center; color:#00E5FF;'>IDS Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to the Intelligent Intrusion Detection System (IDS) Dashboard.  
    Use the sidebar to navigate between:
    - **Anomaly Detection**: See model metrics and detected anomalies  
    - **Visualizations**: ROC, PR, confusion matrix plots  
    - **Explainability (XAI)**: Inspect individual anomalies with SHAP explanations  
    """, unsafe_allow_html=True)
    from Home import render_home
    render_home()
# --- Anomaly Detection Page ---
elif page == "Anomaly Detection":
    from pages.Anomaly_Detection import render_anomaly_detection
    render_anomaly_detection()  # your previous visualization code

# --- Visualizations Page ---
elif page == "Visualizations":
    from pages.Visualizations import render_visualizations
    render_visualizations()  # your previous metrics/ROC/PR plots

# --- Explainability (XAI) Page ---
elif page == "Explainability (XAI)":
    from pages.XAI_visualizations import render_xai
    render_xai()  # the interactive XAI page we built