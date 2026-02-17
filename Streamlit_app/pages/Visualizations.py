import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go


# Page Config
st.set_page_config(
    page_title="Models Visualizations & Ensemble",
    layout="wide"
)

# CSS Styling (neon headers, hologram frame, metric cards)
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
    justify-content: flex-start;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 40px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
    border: 2px solid #00E5FF;
    border-radius: 12px;
    padding: 20px;
    width: 480px;
    height: 120px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 0 18px rgba(0,229,255,0.18);
    transition: all 0.3s ease;
}
.metric-card h3 { color: #14FFEC; margin-bottom: 6px; }
.metric-card p { font-size: 1rem; margin: 0; }
.metric-card:hover {
    transform: translateY(-6px) scale(1.05);
    border-color: #7F5AF0;
    box-shadow: 0 0 28px rgba(127,90,240,0.25);
}

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

# Paths
BASE_DIR = r"C:/Graduation Project/AI_Cybersecurity"

@st.cache_data
def load_artifacts():
    ae_preds = joblib.load(os.path.join(BASE_DIR, "Models", "ae_preds.pkl"))
    ae_threshold = float(joblib.load(os.path.join(BASE_DIR, "Models", "ae_threshold.pkl")))
    iso_preds = joblib.load(os.path.join(BASE_DIR, "Models", "iso_preds.pkl"))
    y_true = joblib.load(os.path.join(BASE_DIR,"Data", "Processed", "test_labels.pkl"))
    return ae_preds, ae_threshold, iso_preds, y_true

ae_preds, ae_threshold, iso_preds, y_true = load_artifacts()

# Title
# Page Title inside hologram wrapper
st.markdown('<div class="hologram-wrap"><h2>Models Visualizations & Metrics</h2></div>', unsafe_allow_html=True)

# Model Selector
model_choice = st.radio(
    "Select Model",
    ["AutoEncoder", "Isolation Forest", "Ensemble"],
    horizontal=True
)

# Helper function: Metrics & Plots
def display_metrics(y_true, y_pred, probs=None):
    acc = accuracy_score(y_true, y_pred)
    st.markdown(f"**Accuracy:** {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    st.markdown("**Confusion Matrix:**")
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"))
    st.plotly_chart(fig_cm)

    st.markdown("**Classification Report:**")
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    if probs is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig_roc)

        precision, recall, _ = precision_recall_curve(y_true, probs)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
        fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(fig_pr)

# AutoEncoder
if model_choice == "AutoEncoder":
    anomaly_flags = (ae_preds > ae_threshold).astype(int)
    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-card'><h3>Total Samples</h3><p>{len(ae_preds)}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Anomalies Detected</h3><p>{anomaly_flags.sum()}</p></div>", unsafe_allow_html=True)

    fig_hist = px.histogram(ae_preds, nbins=50, marginal="box", title="Reconstruction Error")
    fig_hist.add_vline(x=ae_threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {ae_threshold:.4f}")
    st.plotly_chart(fig_hist)

    st.dataframe(pd.DataFrame({"Reconstruction_Error": ae_preds, "Anomaly": anomaly_flags}).head(100))
    display_metrics(y_true, anomaly_flags, probs=ae_preds)

# Isolation Forest
elif model_choice == "Isolation Forest":
    iso_flags = (iso_preds == -1).astype(int)
    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-card'><h3>Normal Samples</h3><p>{(iso_flags==0).sum()}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Anomalies Detected</h3><p>{iso_flags.sum()}</p></div>", unsafe_allow_html=True)

    fig_count = px.histogram(["Normal" if x==0 else "Anomaly" for x in iso_flags], title="Isolation Forest Predictions")
    st.plotly_chart(fig_count)

    st.dataframe(pd.DataFrame({"Prediction": iso_flags}).head(100))
    display_metrics(y_true, iso_flags)


# Ensemble (AutoEncoder + Isolation Forest)
else:
    ae_flags = (ae_preds > ae_threshold).astype(int)
    iso_flags = (iso_preds == -1).astype(int)
    
    ensemble_flags = ((ae_flags + iso_flags) > 0).astype(int)
    ensemble_probs = (ae_preds/ae_preds.max() + iso_flags) / 2

    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-card'><h3>Total Samples</h3><p>{len(ae_preds)}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Anomalies Detected</h3><p>{ensemble_flags.sum()}</p></div>", unsafe_allow_html=True)

    fig_count = px.histogram(["Normal" if x==0 else "Anomaly" for x in ensemble_flags], title="Ensemble Predictions")
    st.plotly_chart(fig_count)

    st.dataframe(pd.DataFrame({"Ensemble_Prediction": ensemble_flags}).head(100))
    display_metrics(y_true, ensemble_flags, probs=ensemble_probs)

st.markdown("</div>", unsafe_allow_html=True)