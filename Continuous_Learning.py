import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

PROCESSED_PATH = "Data/Processed/"
FEEDBACK_PATH = "Data/feedback.csv"
MODEL_PATH = "Models/isolation_forest.pkl"

WEIGHT_FACTOR = 5


def retrain_models():

    # 1. Check feedback exists

    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback found.")
        return

    feedback_df = pd.read_csv(FEEDBACK_PATH)

    if feedback_df.empty:
        print("Feedback empty.")
        return

    feedback_df.drop_duplicates(inplace=True)
    feedback_df.dropna(inplace=True)

    if "true_label" not in feedback_df.columns:
        raise ValueError("Feedback must contain true_label column")

    # 2. Load datasets

    X_train = joblib.load(PROCESSED_PATH + "train_scaled.pkl")
    y_train = joblib.load(PROCESSED_PATH + "train_labels.pkl")

    X_val = joblib.load(PROCESSED_PATH + "val_scaled.pkl")
    y_val = joblib.load(PROCESSED_PATH + "val_labels.pkl")

    # 3. Separate feedback

    X_feedback = feedback_df.drop("true_label", axis=1)
    y_feedback = feedback_df["true_label"]

    # 4. Weight feedback

    X_feedback_weighted = pd.concat([X_feedback] * WEIGHT_FACTOR)
    y_feedback_weighted = pd.concat([y_feedback] * WEIGHT_FACTOR)

    # 5. Merge safely

    X_combined = pd.concat(
        [pd.DataFrame(X_train), X_feedback_weighted],
        ignore_index=True
    )

    y_combined = pd.concat(
        [pd.Series(y_train), y_feedback_weighted],
        ignore_index=True
    )

    # 6. Evaluate old model

    old_model = joblib.load(MODEL_PATH)

    old_preds = old_model.predict(X_val)
    old_preds = np.where(old_preds == -1, 1, 0)
    old_f1 = f1_score(y_val, old_preds)

    # 7. Retrain new model

    new_model = IsolationForest(contamination=0.05)
    new_model.fit(X_combined)

    new_preds = new_model.predict(X_val)
    new_preds = np.where(new_preds == -1, 1, 0)
    new_f1 = f1_score(y_val, new_preds)

    print("Old F1:", old_f1)
    print("New F1:", new_f1)

    # 8. Replace if better
   
    if new_f1 > old_f1:
        joblib.dump(new_model, MODEL_PATH)
        print("Model updated successfully.")
    else:
        print("Retraining rejected (performance worse).")