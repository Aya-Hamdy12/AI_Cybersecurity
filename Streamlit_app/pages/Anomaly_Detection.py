import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

st.title("Anomaly Detection")

# Load pre-trained model
try:
    with open("Models/iso_preds.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please add 'iso_preds.pkl' in the Models folder.")
    st.stop()

# Upload CSV for anomaly detection
uploaded_file = st.file_uploader("Upload CSV file for anomaly detection", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    if st.button("Detect Anomalies"):
        try:
            predictions = model.predict(df)
            df['Anomaly'] = predictions
            st.write("Detection Results:")
            st.dataframe(df)
            st.write("Legend: 1 = Normal, -1 = Anomaly")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
