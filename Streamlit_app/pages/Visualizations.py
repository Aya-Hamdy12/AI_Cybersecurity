import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CSS / Styling for hologram look
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');

    * { font-family: 'Rajdhani', sans-serif; }
    body, .main { background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%); color: #E0E0E0; }

    h1,h2,h3 { font-family: 'Orbitron', monospace !important; text-transform: uppercase; letter-spacing: 2px; }
    h1 { color: #00E5FF; text-shadow: 0 0 12px rgba(0,229,255,0.45); font-weight:900; animation: fadeIn 0.6s ease-out; }
    @keyframes fadeIn { from { opacity:0; transform: translateY(-10px);} to {opacity:1; transform:translateY(0);} }

    .metric-card {
        background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
        border: 2px solid #00E5FF;
        padding: 20px;
        border-radius: 12px;
        text-align:center;
        transition: all 0.35s ease;
        box-shadow: 0 0 18px rgba(0,229,255,0.18);
    }
    .metric-card:hover { transform: translateY(-6px) scale(1.02); border-color:#7F5AF0; box-shadow: 0 0 28px rgba(127,90,240,0.18); }
    .metric-card h1 { color:#14FFEC; font-size:1.8rem; margin:6px 0; }

    .stButton > button {
        background: linear-gradient(135deg, #14FFEC 0%, #7F5AF0 100%);
        border: none;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight:700;
        letter-spacing:1px;
        transition: transform .2s;
    }
    .stButton > button:hover { transform: translateY(-3px); }

    [data-testid="stSidebar"] {
        background: rgba(10,15,31,0.95);
        border-right: 2px solid rgba(20,255,236,0.06);
        padding-top:8px;
    }
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255,255,255,0.02);
        padding:8px 12px;
        border-radius:6px;
        color:#bff7f6;
    }
    .stDataFrame { background: rgba(15,20,40,0.65); border-radius:6px; }

    .hologram-wrap {
        width: 100%;
        max-width: 1080px;
        margin: 20px auto;
        position: relative;
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(20,255,236,0.03), rgba(127,90,240,0.02));
        border: 1px solid rgba(20,255,236,0.06);
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        display:flex;
        justify-content:center;
        align-items:center;
        flex-direction: column;
    }

    .holo-frame {
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        padding: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        box-shadow: inset 0 0 60px rgba(20,255,236,0.02);
    }

    @media (max-width:900px) { .hologram-wrap { padding: 12px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2>Visualizations</h2>", unsafe_allow_html=True)

# Load data
data = pd.read_csv("C:/Graduation Project/AI_Cybersecurity/cicids2017.csv")

st.write("Data preview:")
st.dataframe(data.head())

# Example visualization
# st.write("Histogram of a numeric column")
# fig, ax = plt.subplots()
# sns.histplot(data['column_name'], bins=30, kde=True, ax=ax)
# st.pyplot(fig)
