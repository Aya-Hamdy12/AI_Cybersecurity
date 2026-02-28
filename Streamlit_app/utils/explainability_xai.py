
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import shap
import streamlit as st
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models_path = os.path.join(project_root, "Models")
data_path   = os.path.join(project_root, "Data", "Processed")
X_test  = joblib.load(os.path.join(data_path, "test_scaled.pkl"))
X_train = joblib.load(os.path.join(data_path, "train_scaled.pkl"))
scaler  = joblib.load(os.path.join(data_path, "scaler_cicids.pkl"))

# Load models
iso_model = joblib.load(os.path.join(models_path, "iso_model.pkl"))
ae_model  = load_model(os.path.join(models_path, "autoencoder.keras"))
pca_model = joblib.load(os.path.join(models_path, "pca_model.pkl"))  # NEW

# Load ensemble predictions
ensemble_df = pd.read_csv(os.path.join(data_path, "ensemble_preds.csv"))
ensemble_threshold = 0.879759519038076

# Background data for SHAP

np.random.seed(42)
background_size = 50
background_indices = np.random.choice(X_train.shape[0], background_size, replace=False)
background_data = X_train[background_indices]

shap.initjs()  # SHAP JS visualization setup

# SHAP Explainers
masker = shap.maskers.Independent(background_data)

# Isolation Forest Explainer
def iso_wrapper(X):
    return iso_model.decision_function(X)
iso_explainer = shap.Explainer(iso_wrapper, masker, algorithm="permutation")

# Autoencoder Explainer
def ae_pred_func(X):
    recon = ae_model.predict(X, verbose=0)
    mse = np.mean((X - recon)**2, axis=1)
    return mse
ae_explainer = shap.KernelExplainer(ae_pred_func, background_data)

# PCA Explainer (reconstruction error)
def pca_pred_func(X):
    recon = pca_model.inverse_transform(pca_model.transform(X))
    mse = np.mean((X - recon)**2, axis=1)
    return mse
pca_explainer = shap.KernelExplainer(pca_pred_func, background_data)

print("SHAP explainers initialized for ISO, AE, and PCA.")

# SHAP computation

def compute_shap_values(x_sample):
    iso_shap = iso_explainer(x_sample).values
    ae_shap = np.array(ae_explainer.shap_values(x_sample, nsamples=50))
    pca_shap = np.array(pca_explainer.shap_values(x_sample, nsamples=50))
    return iso_shap, ae_shap, pca_shap

# Global cache
shap_cache = {}

def get_or_compute_shap(sample_index, x_sample):
    """Return cached SHAP values or compute if missing"""
    if sample_index in shap_cache:
        return shap_cache[sample_index]
    iso_shap, ae_shap, pca_shap = compute_shap_values(x_sample)
    shap_cache[sample_index] = (iso_shap, ae_shap, pca_shap)
    return iso_shap, ae_shap, pca_shap

# Ensemble SHAP computation

def compute_ensemble_shap(iso_shap, ae_shap, pca_shap, w_iso=0.5, w_ae=0.3, w_pca=0.2):
    ensemble_shap = w_iso * iso_shap + w_ae * ae_shap + w_pca * pca_shap
    model_contrib = {"ISO": w_iso, "AE": w_ae, "PCA": w_pca}  # optional
    return ensemble_shap, model_contrib

# def compute_ensemble_shap(iso_shap, ae_shap, pca_shap, w_iso=0.5, w_ae=0.3, w_pca=0.2):
#     ensemble_shap = w_iso * iso_shap + w_ae * ae_shap + w_pca * pca_shap
#     model_contrib = {
#         "ISO": w_iso * iso_shap,
#         "AE": w_ae * ae_shap,
#         "PCA": w_pca * pca_shap
#     }
    return ensemble_shap, model_contrib
# Feature names
feature_names_file = os.path.join(project_root, "feature_names.json")
if os.path.exists(feature_names_file):
    columns_names = json.load(open(feature_names_file))
else:
    columns_names = [
        "bwd_packet_length_std","flow_iat_std","fwd_iat_max","flow_iat_max","flow_duration",
        "fwd_iat_total","fwd_iat_mean","destination_port","fwd_iat_std","packet_length_variance",
        "max_packet_length","bwd_iat_total","flow_iat_min","bwd_iat_mean","bwd_iat_max",
        "bwd_iat_std","bwd_packets_s","flow_iat_mean","bwd_packet_length_max","flow_bytes_s",
        "fwd_iat_min","fwd_packet_length_std","packet_length_std","total_fwd_packets","idle_max",
        "idle_mean","idle_min","active_min","active_max","active_mean","total_length_of_fwd_packets",
        "subflow_fwd_bytes","bwd_header_length","fwd_header_length","fwd_packet_length_max",
        "packet_length_mean","bwd_packet_length_mean","bwd_iat_min","average_packet_size",
        "fwd_packets_s","flow_packets_s","act_data_pkt_fwd","fwd_packet_length_mean","bwd_packet_length_min",
        "fin_flag_count","fwd_packet_length_min","min_packet_length","min_seg_size_forward",
        "ack_flag_count","psh_flag_count"
    ]
    json.dump(columns_names, open(feature_names_file, "w"))

# Rank top features
def rank_top_features(ensemble_shap_values, feature_names, top_n=5):
    ensemble_vals_flat = ensemble_shap_values[0]
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": ensemble_vals_flat,
        "Absolute Impact": np.abs(ensemble_vals_flat)
    })
    return df.sort_values(by="Absolute Impact", ascending=False).head(top_n)

# Text explanation

def generate_explanation(top_features):
    explanation = "This traffic was flagged as anomalous primarily due to:\n"
    for _, row in top_features.iterrows():
        direction = "increased anomaly score" if row["SHAP Value"] > 0 else "reduced anomaly score"
        explanation += f"- {row['Feature']} ({row['SHAP Value']:.4f}) → {direction}\n"
    return explanation

# Waterfall plot

def plot_ensemble_waterfall(
    x_sample, ensemble_shap_values, feature_names, base_value=0, show_figure=True, platform="streamlit"
):
    """Plot SHAP waterfall for a single sample, compatible with Streamlit"""
    shap_vals = ensemble_shap_values[0]
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=x_sample[0],
        feature_names=feature_names
    )

    # Create a figure explicitly
    fig, ax = plt.subplots(figsize=(10,6))
    shap.plots.waterfall(explanation, show=False)
    
    if show_figure:
        if platform == "streamlit":
            st.pyplot(fig)  # pass the figure
        else:
            plt.show(fig)
    
    return fig

# Input validation
def validate_and_prepare_input(x_sample, expected_num_features):
    if isinstance(x_sample, (pd.DataFrame, pd.Series)):
        x_sample = x_sample.values
    x_sample = np.array(x_sample)
    if x_sample.size == 0:
        raise ValueError("Input sample is empty.")
    if x_sample.ndim == 1:
        x_sample = x_sample.reshape(1, -1)
    if x_sample.ndim > 2:
        x_sample = np.squeeze(x_sample)
    if x_sample.ndim == 1:
        x_sample = x_sample.reshape(1, -1)
    if x_sample.shape[1] != expected_num_features:
        raise ValueError(f"Feature mismatch: expected {expected_num_features}, got {x_sample.shape[1]}")
    return x_sample.astype(np.float32)

def explain_single_anomaly(sample_index, top_n=5, show_figure=True):

    x_sample = validate_and_prepare_input(
        X_test[sample_index:sample_index+1],
        X_train.shape[1]
    )

    iso_shap, ae_shap, pca_shap = get_or_compute_shap(sample_index, x_sample)

    # FIRST compute ensemble
    ensemble_shap, model_contrib = compute_ensemble_shap(
        sample_index,
        iso_shap,
        ae_shap,
        pca_shap
    )

    # ensemble_shap, model_contrib = compute_ensemble_shap(iso_shap, ae_shap, pca_shap)
    # st.write(model_contrib)

    # THEN rank features
    top_features = rank_top_features(
        ensemble_shap,
        columns_names,
        top_n=top_n
    )

    explanation_text = generate_explanation(top_features)

    waterfall_fig = plot_ensemble_waterfall(
        x_sample,
        ensemble_shap,
        columns_names,
        show_figure=show_figure,
        platform="streamlit"
    )

    return top_features, explanation_text, waterfall_fig, model_contrib

