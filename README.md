🛡️ Sentinel-AI
Hybrid Network Intrusion Prevention System (NIDS)

🚀 Overview
Sentinel-AI is an advanced Hybrid Network Intrusion Prevention System (NIDS) designed to defend modern network infrastructures against known and zero-day cyber threats.
Unlike traditional rule-based firewalls, Sentinel-AI employs a 2-Stage Hybrid Ensemble Learning Architecture, combining Generative AI and classical Machine Learning to achieve high precision with minimal false positives.
🔥 92% Accuracy | 98% Precision | Intelligent Self-Healing System

🧠 Key Innovation
🔐 Generative AI as a “Gatekeeper”
Sentinel-AI uses a GAN-based anomaly detector trained exclusively on normal network traffic, enabling it to detect previously unseen (zero-day) attacks.
🤝 Ensemble Intelligence
Suspicious traffic is verified by a Voting Committee of 5 ML models, ensuring reliable attack confirmation before action is taken.

## ⚙️ System Architecture & Tech Stack
| Component | Technology |
|---------|-----------|
| Core Language | [Python 3.12](https://www.python.org/downloads/release/python-3120/) |
| Deep Learning | [PyTorch](https://pytorch.org/) (GAN-based Anomaly Detection) |
| ML Framework | [Scikit-Learn](https://scikit-learn.org/stable/) |
| Data Processing | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Ensemble Logic | [Voting Classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) |
| OS Environment | [Linux (Ubuntu 24.04)](https://ubuntu.com/download/desktop) |

🛠️ 2-Stage Detection Pipeline

🧩 Stage 1 — The Gatekeeper (GAN)
Trained only on benign traffic
Detects anomalies & zero-day threats
Flags suspicious sessions for deeper inspection

🔍 Stage 2 — Investigator Committee
A voting ensemble of:
Random Forest
SGD Classifier
K-Nearest Neighbors
Gaussian Naive Bayes
Decision Tree
Only consensus-verified threats are classified as attacks.

🚨 Stage 3 — Active Response Engine
Tracks attacker Session IDs
Automatically blocks users crossing a strike threshold
Prevents repeated malicious attempts

📊 Performance & Evaluation
📌 Evaluated on a real-world cybersecurity dataset containing 9,537 network traffic samples.

🏆 Final Model Performance
Metric	Score	Insight
Accuracy	92%	High reliability in traffic classification
Precision	98%	Extremely low false positives
Recall	84%	Effective attack capture rate
Threats Blocked	1,737	Malicious sessions automatically stopped
Self-Healing Samples	3,325	False positives retained for retraining

📉 Classification Report (Hybrid Committee)
              precision    recall  f1-score   support

Normal (0)       0.88      0.99      0.93      2960
Attack (1)       0.99      0.84      0.91      2376

Accuracy                             0.92      5336

💻 How to Run (Linux / Ubuntu)
1️⃣ Clone the Repository
git clone https://github.com/YOUR-USERNAME/Intrusion-Detection-System.git
cd Intrusion-Detection-System

2️⃣ Install Dependencies
pip install pandas numpy scikit-learn torch tqdm

3️⃣ Train the Models
(Generates all 6 AI models & processors)
python3 train_models.py

4️⃣ Run Detection Simulation
python3 detect_final_demo.py

🔮 Future Roadmap (Semester 6)
🌐 Real-Time Network Integration using Scapy
☁️ Cloud Deployment with Docker & AWS
🔄 Automated Daily Retraining for adaptive threat learning
