import numpy as np
import joblib

class EnsembleIDS:
    def __init__(self, iso_model, ae_model, pca_model,
                 ae_threshold, pca_threshold,
                 w_iso=0.33, w_ae=0.33, w_pca=0.34):
        self.iso_model = iso_model
        self.ae_model = ae_model
        self.pca_model = pca_model
        self.ae_threshold = ae_threshold
        self.pca_threshold = pca_threshold
        self.w_iso = w_iso
        self.w_ae = w_ae
        self.w_pca = w_pca

    def _normalize_scores(self, scores, train_scores, invert=False):
        """
        Convert raw scores into percentiles based on training scores.
        """
        if invert:
            scores = -scores
        percentiles = np.searchsorted(np.sort(train_scores), scores) / len(train_scores)
        return np.clip(percentiles, 0, 0.99)

    def predict(self, X, iso_train_scores, ae_train_scores, pca_train_scores):
        # Individual predictions
        iso_raw = self.iso_model.predict(X)
        iso_pred = np.where(iso_raw == -1, 1, 0)  # anomaly = 1

        ae_recon = self.ae_model.predict(X)
        ae_error = np.mean((X - ae_recon)**2, axis=1)
        ae_pred = (ae_error > self.ae_threshold).astype(int)

        pca_proj = self.pca_model.transform(X)
        pca_recon = self.pca_model.inverse_transform(pca_proj)
        pca_error = np.mean((X - pca_recon)**2, axis=1)
        pca_pred = (pca_error > self.pca_threshold).astype(int)

        # Percentile normalization for weighted ensemble
        iso_score = self._normalize_scores(iso_raw, iso_train_scores, invert=True)
        ae_score  = self._normalize_scores(ae_error, ae_train_scores)
        pca_score = self._normalize_scores(pca_error, pca_train_scores)

        # Weighted ensemble
        weighted_score = self.w_iso*iso_score + self.w_ae*ae_score + self.w_pca*pca_score
        ensemble_pred = (weighted_score >= 0.5).astype(int)  # threshold can be adjusted

        return {
            "iso_pred": iso_pred,
            "ae_pred": ae_pred,
            "pca_pred": pca_pred,
            "weighted_score": weighted_score,
            "ensemble_pred": ensemble_pred
        }
