import streamlit as st
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="AI Cybersecurity App",
    layout="wide",
)

# ---------------------------
# CSS Styling (neon headers, hologram frame, metric cards)
# ---------------------------
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
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 40px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
    border: 2px solid #00E5FF;
    border-radius: 12px;
    padding: 25px 20px;
    width: 220px;
    text-align: center;
    box-shadow: 0 0 18px rgba(0,229,255,0.18);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.05);
    border-color: #7F5AF0;
    box-shadow: 0 0 28px rgba(127,90,240,0.25);
}

.metric-card h3 { color: #14FFEC; }
.metric-card p { font-size: 0.9rem; }

.stImage img { border-radius: 12px; }

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

# ---------------------------
# Hologram frame with title + image
# ---------------------------
# st.markdown('<h1>AI-Powered Cybersecurity Anomaly Detection</h1>', unsafe_allow_html=True)

# Load image (make sure image.png is in the same folder)
image = Image.open("image.png")
st.image(image, caption="", width=1080)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Metric Cards Section
# ---------------------------
st.markdown('<h2>REAL-TIME CYBERSECURITY MONITORING</h2>', unsafe_allow_html=True)
st.markdown("""
<div class="metric-container">
    <div class="metric-card">
        <h3>DATASET</h3>
        <p>CICIDS2017</p>
        <p>53 features  -  2520751 Records</p>
    </div>
    <div class="metric-card">
        <h3>MODELS</h3>
        <p>Isolation Forest / Autoencoder</p>
        <p>With Ensemble</p>
    </div>
    <div class="metric-card">
        <h3>BEST MODEL</h3>
        <p>Isolation Forest</p>
        <p> 94% acc</p>
    </div>
    <div class="metric-card">
        <h3>ALERTS</h3>
        <p>REAL-TIME</p>
        <p>CSV logging</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------
# Project Description
# ---------------------------
st.markdown('<h2>Project Description</h2>', unsafe_allow_html=True)
st.write("""
This project aims to design and develop an **AI-powered cybersecurity system** that detects network anomalies 
and potential cyberattacks in real-time. Traditional Intrusion Detection Systems (IDS) rely on fixed rule sets 
and signatures, making them ineffective against zero-day attacks and insider threats.

Our system introduces a more adaptive approach by integrating **Artificial Intelligence (AI)**, **Explainable AI (XAI)**, 
and **continuous learning techniques**. The AI component automatically learns each organization’s unique patterns 
of normal network behavior and identifies deviations as potential anomalies.

Through an **ensemble of models**, the system achieves stable and reliable anomaly detection. A **self-healing mechanism** 
monitors performance over time and triggers retraining when accuracy drops, allowing the model to adapt to evolving network conditions.

The system is designed as an **API-based service**, allowing integration with existing monitoring tools or signature-based IDS 
for a hybrid setup. The **XAI layer** explains the reasoning behind each alert in a human-understandable way, enabling 
security analysts to trust, interpret, and act on system outputs.

Traditional IDS are efficient for known threats; our system complements that by extending protection to **unknown and evolving threats**. 
Efficiency can improve with optimization, but **adaptability and coverage** are our focus.
""")

st.header("Goal")
st.markdown("""
Build an intelligent, explainable, and continuously learning system that can monitor network activity and identify unusual behavior 
without relying solely on predefined attack signatures. The system focuses on **adaptability**, **per-organization learning**, and **resilience**, 
complementing traditional IDS by extending protection to zero-day, insider, and evolving cyber threats.
""")

st.header("Project Scope")
st.markdown("""
- Develop a **hybrid IDS** combining signature-based detection with AI-driven anomaly detection.  
- Implement **ensemble-based AI models** to improve stability and accuracy.  
- Integrate **Explainable AI (XAI) techniques** — SHAP/LIME — for clear, human-understandable explanations.  
- Introduce a **self-healing mechanism** that triggers retraining automatically.  
- Enable **organization-specific learning** for continuous adaptation.  
- Design the system as an **API-based service**.  
- Implement a **web-based dashboard** for real-time visualization of detections, risk levels, and explanations.  
- Use public network traffic datasets (CICIDS2017, UNSW-NB15) to simulate realistic environments.
""")

st.header("Project Limitations")
st.markdown("""
- The system will use **simulated network traffic** from public datasets rather than capturing live enterprise data.  
- Signature-based detection relies on open-source IDS tools (Snort/Suricata) with default or customized rules.  
- Focus on **network-based detection only**.  
- Real-time monitoring will be simulated via data streaming.  
- Deployment is limited to **proof-of-concept** scale.  
- Basic security features included; enterprise-grade security is beyond the project scope.
""")

st.header("Project Deliverables")
st.markdown("""
- **Hybrid IDS**: Signature + AI anomaly detection.  
- **AI Anomaly Detection Model**: Ensemble models detecting deviations.  
- **Explainability Module**: SHAP/LIME integration for transparent results.  
- **Self-Healing & Continuous Learning**: Automated retraining and performance monitoring.  
- **API-Based Backend**: Cloud-hosted API connecting all modules.  
- **Web-Based Dashboard**: Real-time visualization of alerts and explanations.  
- **Simulation Environment**: Dataset-based network traffic simulator.  
- **Deployment Setup**: Cloud deployment for online access.  
- **Documentation & Reports**
""")

st.header("Project Methodology / System Pipeline")
st.markdown("""
1. **Data Acquisition and Preparation**  
   - Datasets: CICIDS2017  
   - Tools: Python (Pandas, NumPy, Scikit-learn)  
   - Tasks: Merge CSVs, feature selection, missing values handling, encoding, normalization, train-test split  

2. **AI-Based Anomaly Detection Models**  
   - Models: Autoencoder, Isolation Forest, One-Class SVM  
   - Ensemble for more stable predictions  
   - Continuous learning with periodic retraining  

3. **Signature-Based IDS Integration**  
   - Tools: Snort / Suricata  
   - Acts as pre-processing filter before AI analysis  

4. **Explainability Layer (XAI)**  
   - Techniques: SHAP, LIME  
   - Generates feature-importance and visual plots  

5. **Self-Healing and Continuous Learning**  
   - Monitor model performance  
   - Retraining triggered automatically when metrics drop  

6. **Backend API Service**  
   - RESTful endpoints: /upload, /predict, /retrain, /report  
   - Handles multiple organizations and stores logs  

7. **Database Layer**  
   - Store predictions, explanations, timestamps, retraining data  
   - PostgreSQL or MongoDB  

8. **Frontend Dashboard**  
   - Tools: Streamlit or Angular  
   - Real-time anomaly display, risk levels, and XAI visuals  

9. **Simulation and Testing**  
   - Feeder script streams dataset records  
   - Validate full pipeline and test API endpoints  

10. **Deployment**  
    - Docker, GitHub, cloud deployment  
    - Enables real-time simulation or live network monitoring
""")

# st.header("Benchmark Studies: Traditional vs. AI-Based IDS")
# st.markdown("""
# 1. **Investigation of Machine Learning Algorithms for Network Intrusion Detection (2022)**  
#    - Signature-based detection: Detection rate 92.73%, False positive 0.54%  
#    - Anomaly-based detection: Detection rate 99.0%, False positive 12.6%  
#    - Hybrid model: Detection rate 98.3%, False positive 1.5%  

# 2. **Self-Healing Hybrid Intrusion Detection System: Ensemble ML Approach (2024)**  
#    - Signature-only: True positive rate ~91%, False positive ~3%  
#    - Hybrid ensemble: True positive rate ~97%, False positive ~8%
# """)