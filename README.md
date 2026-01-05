# 🛡️ Sentinel-AI: Hybrid Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge)
![Security](https://img.shields.io/badge/Domain-Cyber_Security-red?style=for-the-badge)

## 📖 Executive Summary
**Sentinel-AI** is an advanced Network Intrusion Detection System (NIDS) engineered to secure network infrastructures against modern cyber threats. Unlike traditional firewalls that rely on static rules, this system utilizes a **Hybrid Ensemble Learning Architecture**.

By aggregating the predictive power of five distinct algorithms—**Random Forest, Decision Tree, KNN, Naive Bayes, and SGD**—the system achieves a robust **93% accuracy rate**, significantly reducing false positives and ensuring high-fidelity threat detection.

---

## ⚙️ Technical Architecture & Tech Stack

The system is built on a modular data science pipeline.

| Component | Technology Used |
| :--- | :--- |
| **Core Language** | Python 3.x |
| **ML Framework** | Scikit-Learn (sklearn) |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Ensemble Logic** | Voting Classifier (Soft Voting) |
| **Environment** | Linux |

---

## 📊 Performance Analysis & Results

The model was rigorously tested on a dataset comprising **8,365 network traffic samples**. The **Hybrid Committee Model** outperformed individual classifiers in reliability.

### 🏆 Final Model Evaluation
| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Accuracy** | **93.0%** | High reliability in distinguishing normal vs. attack traffic. |
| **Precision** | **0.99** | Extremely low False Positive Rate (Only 1% error in flagging attacks). |
| **F1-Score** | **0.91** | Balanced performance between Precision and Recall. |
| **Threats Blocked**| **2,768** | Successfully identified malicious packets during testing. |

### 📉 Detailed Classification Report (Hybrid Committee)
```text
[REPORT FOR: HYBRID IPS COMMITTEE]
              precision    recall  f1-score   support

           0       0.89      0.99      0.94      4628
           1       0.99      0.84      0.91      3737

    accuracy                           0.93      8365

## 💻 How to Run This Project (Linux/Ubuntu)

Since this project is designed for a Linux environment, follow these steps to run the detection system:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/Intrusion-Detection-System.git](https://github.com/Mann65121/Intrusion-Detection-System.git)
   cd Intrusion-Detection-System
