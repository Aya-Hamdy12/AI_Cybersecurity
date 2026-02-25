import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


project_root = os.path.abspath(os.path.join(os.getcwd()))
models_path = os.path.join(project_root, "Models")
data_path   = os.path.join(project_root, "Data", "Processed")

X_test = joblib.load(os.path.join(data_path, "test_scaled.pkl"))
X_train = joblib.load(os.path.join(data_path, "train_scaled.pkl"))
scaler = joblib.load(os.path.join(data_path, "scaler_cicids.pkl"))

iso_model = joblib.load(os.path.join(models_path, "iso_model.pkl"))
ae_model = load_model(os.path.join(models_path, "autoencoder.keras"))


iso_scores = joblib.load(os.path.join(models_path, "iso_test_scores.pkl"))
ae_recon_errors = joblib.load(os.path.join(models_path, "ae_preds.pkl"))

ensemble_threshold = 0.879759519038076
ensemble_df = pd.read_csv(os.path.join(data_path, "ensemble_preds.csv"))


# Create background data (a baseline reference for SHAP explanations)
np.random.seed(42)
background_size = 50
background_indices = np.random.choice(X_train.shape[0], background_size, replace=False)
background_data = X_train[background_indices]


import shap
from shap import KernelExplainer
shap.initjs()  #shap js visualization setup


# Create/initialize SHAP Explainers

# SHAP Explainer for Isolation Forest (tree-based)

# create masker from background data
masker = shap.maskers.Independent(background_data)
def iso_wrapper(X):
    return iso_model.decision_function(X)

iso_explainer = shap.Explainer(iso_wrapper, masker, algorithm="permutation")


# SHAP Explainer for Autoencoder (kernel-based)
def ae_pred_func(x):
    recon = ae_model.predict(x, verbose=0)
    mse = np.mean((x - recon)**2, axis=1)
    return mse

ae_explainer = shap.KernelExplainer(ae_pred_func, background_data)  # use a smaller subset for kernel explainer

print("SHAP explainers initialized.")




def compute_shap_values(x_sample):
    #isolation Forest SHAP values
    iso_shap_values = iso_explainer(x_sample)
    #autoencoder SHAP values
    ae_shap_values = ae_explainer.shap_values(x_sample, nsamples=50)
    # Extract raw arrays safely
    iso_shap_values = iso_shap_values.values          # from Explainer
    ae_shap_values = np.array(ae_shap_values)         # from KernelExplainer
    return iso_shap_values, ae_shap_values  



def compute_ensemble_shap(iso_shap_values, ae_shap_values, w_iso=0.6, w_ae=0.4):
    ensemble_shap_values = w_iso * iso_shap_values + w_ae * ae_shap_values
    return ensemble_shap_values



# create a json file with column names
import os, json

feature_names_file = "feature_names.json"

columns_names = json.load(open(feature_names_file)) if os.path.exists(feature_names_file) else json.dump([
    "bwd_packet_length_std", "flow_iat_std", "fwd_iat_max", "flow_iat_max", "flow_duration", "fwd_iat_total",
    "fwd_iat_mean", "destination_port", "fwd_iat_std", "packet_length_variance", "max_packet_length",
    "bwd_iat_total", "flow_iat_min", "bwd_iat_mean", "bwd_iat_max", "bwd_iat_std", "bwd_packets_s",
    "flow_iat_mean", "bwd_packet_length_max", "flow_bytes_s", "fwd_iat_min", "fwd_packet_length_std",
    "packet_length_std", "total_fwd_packets", "idle_max", "idle_mean", "idle_min", "active_min", "active_max",
    "active_mean", "total_length_of_fwd_packets", "subflow_fwd_bytes", "bwd_header_length", "fwd_header_length",
    "fwd_packet_length_max", "packet_length_mean", "bwd_packet_length_mean", "bwd_iat_min", "average_packet_size",
    "fwd_packets_s", "flow_packets_s", "act_data_pkt_fwd", "fwd_packet_length_mean", "bwd_packet_length_min",
    "fin_flag_count", "fwd_packet_length_min", "min_packet_length", "min_seg_size_forward", "ack_flag_count",
    "psh_flag_count"
], open(feature_names_file, "w"))



# Flatten and rank top feature
def rank_top_features(ensemble_shap_values, feature_names, top_n=5):
    # Flatten for single sample
    ensemble_vals_flat = ensemble_shap_values[0]  # 1D array of SHAP values for the sample
    
    # Create a DataFrame
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": ensemble_vals_flat,
        "Absolute Impact": np.abs(ensemble_vals_flat)
    })
    
    # Sort by absolute impact descending
    feature_importance = feature_importance.sort_values(by="Absolute Impact", ascending=False)
    
    return feature_importance.head(top_n)



def generate_explanation(top_features):
    explanation = "This traffic was flagged as anomalous primarily due to:\n"
    for _, row in top_features.iterrows():
        direction = "increased anomaly score" if row["SHAP Value"] > 0 else "reduced anomaly score"
        explanation += f"- {row['Feature']} ({row['SHAP Value']:.4f}) → {direction}\n"
    return explanation



def plot_ensemble_waterfall(x_sample, ensemble_shap_values, feature_names, base_value=0):
    # Flatten the SHAP values for a single sample
    shap_vals_flat = ensemble_shap_values[0]
    
    explanation = shap.Explanation(
        values=shap_vals_flat,
        base_values=base_value,
        data=x_sample[0],
        feature_names=feature_names
    )
    fig = plt.figure()
    shap.plots.waterfall(explanation, show=False)
    return fig


def validate_and_prepare_input(x_sample, expected_num_features):
    """
    Ensures input is:
    - numpy array
    - numeric
    - shape (1, n_features)
    - correct number of features

    If possible, reshapes automatically.
    Raises clear error if impossible.
    """

    # Convert pandas → numpy
    if isinstance(x_sample, pd.DataFrame) or isinstance(x_sample, pd.Series):
        x_sample = x_sample.values

    # Convert to numpy array
    x_sample = np.array(x_sample)

    # If empty → stop immediately
    if x_sample.size == 0:
        raise ValueError("Input sample is empty. No features were passed.")

    # If 1D (50,) → reshape to (1, 50)
    if x_sample.ndim == 1:
        x_sample = x_sample.reshape(1, -1)

    # If shape is (n, 1, 50) or weird → squeeze
    if x_sample.ndim > 2:
        x_sample = np.squeeze(x_sample)

    # Final reshape check
    if x_sample.ndim == 1:
        x_sample = x_sample.reshape(1, -1)

    # Check feature count
    if x_sample.shape[1] != expected_num_features:
        raise ValueError(
            f"Feature mismatch: Model expects {expected_num_features} features "
            f"but received {x_sample.shape[1]}"
        )

    # Ensure numeric
    x_sample = x_sample.astype(np.float32)

    return x_sample





def explain_single_anomaly(x_sample):
    # Validate & fix input
    x_sample = validate_and_prepare_input(
        x_sample,
        expected_num_features=len(columns_names)
    )
    # Compute SHAP values
    iso_shap, ae_shap = compute_shap_values(x_sample)
    # Compute ensemble SHAP
    ensemble_shap = compute_ensemble_shap(iso_shap, ae_shap, w_iso=0.6, w_ae=0.4)
    # Flatten and rank top features 
    top_features = rank_top_features(ensemble_shap, columns_names, top_n=5)
    # Generate textual explanation
    explanation_text = generate_explanation(top_features)
    print(explanation_text)
    # Plot waterfall
    fig = plot_ensemble_waterfall(x_sample, ensemble_shap, columns_names)
    return top_features, explanation_text, fig
