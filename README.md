<div align="center">

#  AI-Powered Network Intrusion Detection System

### An intelligent, explainable, and continuously learning system for network threat detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-f7931e?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

![RAG](https://img.shields.io/badge/Architecture-RAG_|_Ollama-00ADD8?style=for-the-badge)
![n8n](https://img.shields.io/badge/n8n-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)


</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [AI Models](#-ai-models)
- [XAI — Explainability](#-xai--explainability)
- [Continuous Learning](#-continuous-learning)
- [RAG Chatbot & Automated Alerts](#-rag-chatbot--automated-security-alerts)
- [Dashboard (Streamlit)](#-dashboard-streamlit)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Team](#-team)

---

##  Overview

Traditional Intrusion Detection Systems (IDS) rely on static, predefined signatures to detect threats — making them blind to zero-day attacks, insider threats, and novel malware that has never been seen before.

This project addresses that gap by building an **AI-driven IDS** that:

- Monitors network traffic and detects anomalous behavior using multiple ML/DL models
- Provides **explainable predictions** via XAI techniques (SHAP) so analysts understand *why* a decision was made
- Learns **continuously** to adapt to new and evolving threat patterns
- Exposes findings through an interactive **Streamlit dashboard**
- Includes a **RAG-powered chatbot** (separate branch) that allows security analysts to query threat knowledge in natural language

---

##  Problem Statement

> *Cybersecurity teams are overwhelmed by massive volumes of network events, a large proportion of which are false positives. Signature-based IDS tools miss novel attacks entirely. There is a need for an intelligent system that can learn patterns of normal vs. malicious behavior and explain its reasoning to human analysts.*

This system complements traditional IDS solutions by extending detection coverage to:

- **Zero-day attacks** — threats with no prior signatures
- **Insider threats** — anomalies from trusted internal actors
- **Evolving threats** — attack patterns that drift over time

---

##  System Architecture

The pipeline operates in **two stages**: first detecting whether traffic is anomalous (unsupervised), then classifying the attack type (supervised).

```
┌──────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                             │
│             Network Traffic (CSV / Packet Capture)               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                       │
│      Feature Engineering · Normalization · Class Balancing       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│           STAGE 1 — ANOMALY DETECTION (Unsupervised)             │
│                                                                  │
│   ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│   │   Autoencoder   │  │ Isolation Forest │  │ PCA Recon.    │   │
│   │ (Deep Learning) │  │                  │  │ Error         │   │
│   └────────┬────────┘  └────────┬─────────┘  └──────┬────────┘   │
│            │                   │                    │            │
│            └───────────────────┼────────────────────┘            │
│                                │                                 │
│                                ▼                                 │
│                  ┌─────────────────────────┐                     │
│                  │   Ensemble (Voting)     │                     │
│                  │  Attack Detected? Yes/No│                     │
│                  └────────────┬────────────┘                     │
└───────────────────────────────┼──────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               │ Attack Detected                 │ Benign Traffic
               ▼                                 ▼
┌──────────────────────────────────┐    ┌─────────────────────┐
│  STAGE 2 — ATTACK CLASSIFICATION │    │   No Alert Raised   │
│         (Supervised)             │    └─────────────────────┘
│                                  │
│  ┌──────────┐ ┌────────┐ ┌─────┐ │
│  │  Random  │ │XGBoost │ │Tab  │ │
│  │  Forest  │ │        │ │Net  │ │
│  └──────────┘ └────────┘ └─────┘ │
│         ↓ Best Model Selected ↓  │
│   ┌──────────────────────────┐   │
│   │   Attack Type Output     │   │
│   │ (DDoS / DoS / BruteForce │   │
│   │  / Bot / Web Attack ...) │   │
│   └──────────────────────────┘   │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     XAI EXPLANATION LAYER                        │
│                 SHAP Values · Feature Importance                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS LEARNING MODULE                    │
│         Feedback Loop · Model Retraining · Drift Detection       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
  ┌────────────────┐  ┌──────────────────┐  ┌───────────────────┐
  │   Streamlit    │  │   RAG Chatbot    │  │ Automated Security│
  │   Dashboard    │  │                  │  │ Alerts            │
  └────────────────┘  └──────────────────┘  └───────────────────┘
```

---

##  AI Models

The system uses a **two-stage pipeline**: unsupervised models first detect whether traffic is anomalous, then supervised models classify the specific attack type.

---

### Stage 1 — Anomaly Detection (Unsupervised)

Three models independently analyze network traffic for anomalous patterns. Their outputs are merged into a single **ensemble decision** to determine whether an attack is present.

| Model | Mechanism | Notebook |
|---|---|---|
| **Autoencoder (Deep Learning)** | Learns a compressed representation of normal traffic; reconstruction error above a learned threshold signals an anomaly | `Autoencoder.ipynb` |
| **Isolation Forest** | Isolates outliers using random tree partitioning — anomalous samples require fewer splits to isolate | `Isolation_Forest.ipynb` |
| **PCA Reconstruction Error** | Projects traffic into a lower-dimensional space; high reconstruction deviation indicates abnormal behavior | `PCA_ReconstructionError.ipynb` |

**Ensemble:** The three models are combined via a voting/stacking mechanism (`Ensemble.ipynb`). A sample is flagged as an attack only when a consensus is reached, significantly reducing false positives.

---

### Stage 2 — Attack Classification (Supervised)

When Stage 1 confirms an attack, a supervised classifier identifies the specific attack category (DDoS, DoS, Brute Force, Bot, Web Attack, etc.). Three models were trained and evaluated, with the **best-performing model selected** for production use.

| Model | Strengths | Notebook |
|---|---|---|
| **Random Forest** | Robust, interpretable, handles high-dimensional data well | `RandomForest.ipynb` |
| **XGBoost** | Gradient boosting with strong performance on imbalanced class distributions | `XGBoost.ipynb` |
| **TabNet** | Attention-based deep learning for tabular data; combines high accuracy with built-in feature selection | *(TabNet notebook)* |

>  **Model Selection:** All three supervised models were trained and benchmarked on the CICIDS 2018 dataset. The best-performing model (highest F1-score on the held-out test set) was selected as the final classifier deployed in the system.

---

##  XAI — Explainability

A key differentiator of this system is its **Explainable AI (XAI)** layer, implemented in `explainability_xai.ipynb`.

Because a cybersecurity analyst needs to act on an alert — not just receive a binary flag — every prediction is accompanied by an explanation:

- **SHAP (SHapley Additive exPlanations)** — quantifies the contribution of each network feature to the model's prediction
- **Feature Importance plots** — global view of which traffic characteristics matter most
- Human-readable output that maps feature values to threat reasoning

This bridges the gap between model accuracy and operational trust.

---

##  Continuous Learning

The `Continuous_Learning.py` module enables the system to **adapt over time** without full retraining from scratch.

Key capabilities:
- **Feedback loop** — analyst-confirmed labels can be fed back into the model
- **Incremental retraining** — model weights are updated as new traffic patterns emerge
- **Concept drift detection** — monitors statistical shifts in incoming data distributions

This is critical for keeping the system effective against evolving and novel threats.

---

##  RAG Chatbot & Automated Security Alerts

>  Available in a **separate branch** of this repository.

### RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot has been developed alongside the core detection system. It allows security analysts and investigators to:

- Ask natural language questions about detected threats
- Query internal threat reports and model decisions
- Get context-aware answers grounded in cybersecurity knowledge bases

The RAG pipeline combines a vector store (document retrieval) with a large language model to provide grounded, accurate responses.

### Automated Security Alerts

The system also includes an **automated alerting pipeline** that triggers when the detection engine flags a threat:

- Instant alerts dispatched to the security team upon attack detection
- Alert payloads include: attack type, confidence score, affected traffic features, and SHAP-based explanation
- Designed to integrate with notification channels (email, webhook, SIEM systems)
- Reduces analyst response time by surfacing actionable, context-rich alerts automatically

---

##  Dashboard (Streamlit)

The `Streamlit_app/` folder contains an interactive web dashboard built for both analysts and non-technical stakeholders.

**Features include:**
- Upload network traffic files for analysis
- Visual breakdown of detected attack types
- Per-prediction XAI explanations
- Model performance metrics and confidence scores
- Interactive charts powered by Plotly

>  **Note:** The full-stack web application (frontend + backend API) is currently under active development. The Streamlit app serves as the functional ML interface in the meantime.

---

##  Dataset

**CICIDS 2018** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset (2018)

| Property | Details |
|---|---|
| Source | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| Traffic Types | Benign + 7 attack categories (DDoS, DoS, Brute Force, Infiltration, Bot, Web Attacks, etc.) |
| Format | CSV with 80+ network flow features |
| Preprocessing | `CICIDS_Preprocessing.ipynb` / `New_Preprocessing.ipynb` |

Preprocessing steps include: null handling, feature selection, label encoding, normalization (MinMax / Standard), and class imbalance handling.

---

##  Project Structure



---

##  Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Aya-Hamdy12/AI_Cybersecurity.git
cd AI_Cybersecurity
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard

```bash
cd Streamlit_app
streamlit run app.py
```

### 4. Explore the notebooks

Open any `.ipynb` file in Jupyter Notebook or JupyterLab to explore the model training and evaluation pipelines:

```bash
jupyter notebook
```

---

##  Team

This project was developed as a graduation project by a cross-functional team of 7:

| Name | Role |
|---|---|
| **Ali Emad** | Cybersecurity |
| **Seif Ayman** | Full Stack Development |
| **Ali Azzam** | Full Stack Development |
| **Aya Hamdy** | AI / Machine Learning |
| **Nourhan Osama** | AI / Machine Learning |
| **Salma Ayman** | AI / Machine Learning |
| **Reem Hafez** | AI / Machine Learning |


---

<div align="center">


</div>
