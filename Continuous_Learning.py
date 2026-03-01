# Imports
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model

# Paths
PROCESSED_PATH = "Data/Processed/"
FEEDBACK_PATH = "Data/feedback.csv"
ISO_MODEL_PATH = "Models/iso_model.pkl"
AE_MODEL_PATH = "Models/autoencoder.keras"
PCA_MODEL_PATH = "Models/pca_model.pkl"
ENSEMBLE_MODEL_PATH = "Models/ensemble_model.pkl"

# Hyperparameters
WEIGHT_FACTOR = 5
CONTAMINATION = 0.05
AE_EPOCHS = 20
AE_BATCH_SIZE = 256
W_ISO = 0.45
W_AE = 0.33
W_PCA = 0.22

# Helper functions
##################
def build_autoencoder(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def percentile_normalize(train_scores, test_scores):
    """
    Convert test scores to percentiles based on training scores.
    """
    sorted_train = np.sort(train_scores)
    ranks = np.searchsorted(sorted_train, test_scores)
    percentiles = ranks / len(sorted_train)
    percentiles = np.clip(percentiles, 0, 0.99)
    return percentiles

def find_best_threshold(y_true, scores):
    """
    Find threshold that maximizes F1 score on validation set.
    """
    thresholds = np.linspace(0, 1, 500)
    best_f1 = 0
    best_thresh = 0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def compute_weighted_ensemble(iso_scores, ae_scores, pca_scores,
                              iso_train, ae_train, pca_train, y_val):
    """
    Compute weighted ensemble predictions using percentile normalization.
    """
    iso_percentile = percentile_normalize(iso_train, iso_scores)
    ae_percentile  = percentile_normalize(ae_train, ae_scores)
    pca_percentile = percentile_normalize(pca_train, pca_scores)

    ensemble_score = W_ISO*iso_percentile + W_AE*ae_percentile + W_PCA*pca_percentile
    best_thresh, best_f1 = find_best_threshold(y_val, ensemble_score)
    ensemble_pred = (ensemble_score >= best_thresh).astype(int)

    return ensemble_pred, ensemble_score, best_thresh, best_f1

# Main retraining function

def retrain_models():
    # 1. Load feedback
    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback file found.")
        return
    feedback_df = pd.read_csv(FEEDBACK_PATH)
    if feedback_df.empty:
        print("Feedback file is empty.")
        return
    feedback_df.drop_duplicates(inplace=True)
    feedback_df.dropna(inplace=True)
    if "true_label" not in feedback_df.columns:
        raise ValueError("Feedback must contain 'true_label' column.")

    X_feedback = feedback_df.drop("true_label", axis=1)
    y_feedback = feedback_df["true_label"]

    # 2. Load datasets
    X_train = joblib.load(os.path.join(PROCESSED_PATH, "train_scaled.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_PATH, "train_labels.pkl"))
    X_val = joblib.load(os.path.join(PROCESSED_PATH, "val_scaled.pkl"))
    y_val = joblib.load(os.path.join(PROCESSED_PATH, "val_labels.pkl"))

    # 3. Weight feedback
    X_feedback_weighted = pd.concat([X_feedback]*WEIGHT_FACTOR, ignore_index=True)
    y_feedback_weighted = pd.concat([y_feedback]*WEIGHT_FACTOR, ignore_index=True)

    # 4. Combine datasets
    X_combined = pd.concat([pd.DataFrame(X_train), X_feedback_weighted], ignore_index=True)
    y_combined = pd.concat([pd.Series(y_train), y_feedback_weighted], ignore_index=True)

    # Isolation Forest
    iso_old = joblib.load(ISO_MODEL_PATH) if os.path.exists(ISO_MODEL_PATH) else None
    iso_new = IsolationForest(contamination=CONTAMINATION)
    iso_new.fit(X_combined)

    iso_val_scores = -iso_new.decision_function(X_val)  # higher = more anomalous
    iso_val_preds = np.where(iso_val_scores > np.percentile(-iso_new.decision_function(X_val), 95), 1, 0)

    if iso_old:
        iso_old_preds = np.where(-iso_old.decision_function(X_val) > np.percentile(-iso_old.decision_function(X_val), 95), 1, 0)
        iso_old_f1 = f1_score(y_val, iso_old_preds)
        iso_new_f1 = f1_score(y_val, iso_val_preds)
        if iso_new_f1 > iso_old_f1:
            joblib.dump(iso_new, ISO_MODEL_PATH)
            print(f"Isolation Forest updated (F1: {iso_old_f1:.4f} -> {iso_new_f1:.4f})")
        else:
            print(f"Isolation Forest retraining rejected (F1: {iso_old_f1:.4f} -> {iso_new_f1:.4f})")
    else:
        joblib.dump(iso_new, ISO_MODEL_PATH)
        print(f"Isolation Forest trained (F1: {f1_score(y_val, iso_val_preds):.4f})")

    # Autoencoder
    input_dim = X_combined.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(X_combined, X_combined, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, shuffle=True, verbose=0)

    ae_val_scores = np.mean((X_val - autoencoder.predict(X_val))**2, axis=1)
    ae_val_preds = (ae_val_scores > np.percentile(ae_val_scores, 95)).astype(int)
    autoencoder.save(AE_MODEL_PATH)
    print("Autoencoder retrained and saved.")

    # PCA
    pca = PCA(n_components=min(X_combined.shape))
    pca.fit(X_combined)
    pca_val_scores = np.mean((X_val - pca.inverse_transform(pca.transform(X_val)))**2, axis=1)
    pca_val_preds = (pca_val_scores > np.percentile(pca_val_scores, 95)).astype(int)
    joblib.dump(pca, PCA_MODEL_PATH)
    print("PCA retrained and saved.")

    # Weighted Ensemble
  
    # Load training scores for percentile normalization
    iso_train_scores = -iso_new.decision_function(X_combined)
    ae_train_scores = np.mean((X_combined - autoencoder.predict(X_combined))**2, axis=1)
    pca_train_scores = np.mean((X_combined - pca.inverse_transform(pca.transform(X_combined)))**2, axis=1)

    ensemble_val_preds, ensemble_score, best_thresh, ensemble_f1 = compute_weighted_ensemble(
        iso_val_scores, ae_val_scores, pca_val_scores,
        iso_train_scores, ae_train_scores, pca_train_scores,
        y_val
    )

    print(f"Weighted ensemble F1: {ensemble_f1:.4f}, threshold: {best_thresh:.4f}")

    # Compare with old ensemble
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        old_ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
        old_iso = old_ensemble["isolation_forest"]
        old_ae = old_ensemble["autoencoder"]
        old_pca = old_ensemble["pca"]

        old_iso_scores = -old_iso.decision_function(X_val)
        old_ae_scores = np.mean((X_val - old_ae.predict(X_val))**2, axis=1)
        old_pca_scores = np.mean((X_val - old_pca.inverse_transform(old_pca.transform(X_val)))**2, axis=1)

        old_preds, _, _, old_f1 = compute_weighted_ensemble(
            old_iso_scores, old_ae_scores, old_pca_scores,
            iso_train_scores, ae_train_scores, pca_train_scores,
            y_val
        )
    else:
        old_f1 = 0

    # Save ensemble if better
    if ensemble_f1 > old_f1:
        new_ensemble = {"isolation_forest": iso_new, "autoencoder": autoencoder, "pca": pca}
        joblib.dump(new_ensemble, ENSEMBLE_MODEL_PATH)
        print(f"Weighted ensemble updated (F1: {old_f1:.4f} -> {ensemble_f1:.4f})")
    else:
        print(f"Weighted ensemble retraining rejected (F1: {old_f1:.4f} -> {ensemble_f1:.4f})")

# Run retraining

if __name__ == "__main__":
    retrain_models()