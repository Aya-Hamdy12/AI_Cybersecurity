import joblib
from tensorflow.keras.models import load_model

models_path = r"C:/Graduation Project/AI_Cybersecurity/Models"

iso_model = joblib.load(f"{models_path}/iso_model.pkl")
autoencoder = load_model(f"{models_path}/autoencoder.keras")
pca = joblib.load(f"{models_path}/pca_model.pkl")

# Load thresholds & train scores
ae_threshold = joblib.load(f"{models_path}/ae_threshold.pkl")
pca_threshold = joblib.load(f"{models_path}/pca_threshold.pkl")

iso_train_scores = joblib.load(f"{models_path}/iso_train_scores.pkl")
ae_train_scores  = joblib.load(f"{models_path}/ae_train_scores.pkl")
pca_train_scores = joblib.load(f"{models_path}/pca_train_scores.pkl")

# Create ensemble object
ensemble_obj = EnsembleIDS(
    iso_model=iso_model,
    ae_model=autoencoder,
    pca_model=pca,
    ae_threshold=ae_threshold,
    pca_threshold=pca_threshold,
    w_iso = 0.45,
    w_ae  = 0.33,
    w_pca = 0.22
)

# Save ensemble object
joblib.dump(ensemble_obj, f"{models_path}/ensemble_model.pkl")