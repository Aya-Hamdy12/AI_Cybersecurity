# pages/xai.py
import streamlit as st
import numpy as np
from utils.explainability_xai import explain_single_anomaly

# Page Config
st.set_page_config(page_title="Explainable AI - Attack Analysis", layout="wide")

# =========================
# Custom CSS Styling (Neon headers, hologram frame, metric cards)
# =========================
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
    text-shadow: 0 0 12px rgba(0,229,255,0.45);
}

h1 { font-size: 2.8rem; margin-bottom: 25px; text-align: center; }
h2 { font-size: 2rem; margin-bottom: 20px; text-align: left; }
h3 { font-size: 1.5rem; margin-bottom: 15px; text-align: left; }
h4 { font-size: 1.2rem; margin-bottom: 10px; text-align: left; }
p { color: #b0b0b0; }

.metric-container {
    display: flex;
    justify-content: flex-start;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 30px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
    border: 2px solid #00E5FF;
    border-radius: 12px;
    padding: 20px 15px;
    width: 200px;
    text-align: left;
    box-shadow: 0 0 18px rgba(0,229,255,0.18);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px) scale(1.05);
    border-color: #7F5AF0;
    box-shadow: 0 0 28px rgba(127,90,240,0.25);
}

.metric-card h3 { color: #14FFEC; margin-bottom: 8px; }
.metric-card p { font-size: 0.9rem; }

.hologram-wrap {
    width: 100%;
    max-width: 1000px;
    margin: 0 auto 35px auto;
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(20,255,236,0.03), rgba(127,90,240,0.02));
    border: 1px solid rgba(20,255,236,0.06);
    box-shadow: 0 15px 50px rgba(0,0,0,0.6);
    display:flex;
    flex-direction: column;
    justify-content:center;
    align-items:center;
}
</style>
""", unsafe_allow_html=True)

# Page Title inside hologram wrapper
st.markdown('<div class="hologram-wrap"><h1>Explainable AI</h1></div>', unsafe_allow_html=True)

# Check if attack samples exist
if "attack_samples" not in st.session_state or len(st.session_state.attack_samples) == 0:
    st.info("No attacks detected yet. Run the live stream first.")
    st.stop()

# List of attack times
attack_times = [a["time"] for a in st.session_state.attack_samples]

# Attack Selection
st.subheader("Select Attack to Explain")
selected_time = st.selectbox("Choose detected attack", attack_times)

# Retrieve selected attack
selected_attack = next(a for a in st.session_state.attack_samples if a["time"] == selected_time)
st.markdown(f"Attack detected at: {selected_attack['time']}")

# Get attack index
attack_index = selected_attack["index"]

# Generate explanation
with st.spinner("Generating explanation..."):
    try:
        st.subheader("SHAP Waterfall")
        top_features, explanation_text, fig, model_contrib = explain_single_anomaly(attack_index)

        st.subheader("Explanation")
        st.write(explanation_text)

        # SHAP Waterfall (already generated above)
        # st.pyplot(fig)

        st.subheader("Top Contributing Features")
        st.dataframe(top_features)

    except Exception as e:
        st.error(f"Explanation error: {e}")