🛡️ Sentinel-AI: Hybrid Network Intrusion Detection System (IPS)

📖 Executive Summary

Sentinel-AI is an advanced Network Intrusion Prevention System (NIDS) engineered to secure network infrastructures against modern cyber threats. Unlike traditional firewalls that rely on static rules, this system utilizes a 2-Stage Hybrid Ensemble Learning Architecture.

By combining a Generative AI (GAN) "Gatekeeper" with a Voting Committee of 5 distinct Machine Learning algorithms, the system achieves a robust 92% accuracy rate and 98% precision, significantly reducing false positives and ensuring high-fidelity threat detection.

⚙️ Technical Architecture & Tech Stack

The system is built on a modular data science pipeline.

Component

Technology Used

Core Language

Python 3.12

Deep Learning

PyTorch (for GAN)

ML Framework

Scikit-Learn (sklearn)

Data Processing

Pandas, NumPy

Ensemble Logic

Voting Classifier (Hybrid Committee)

Environment

Linux (Ubuntu 24.04)

🛠️ The 2-Stage Pipeline

Stage 1 (The Gatekeeper): A GAN-based anomaly detector trained only on normal traffic. It flags any deviation (Zero-Day threat) as "Suspicious".

Stage 2 (The Investigator Committee): An ensemble of 5 models (Random Forest, SGD, KNN, Gaussian NB, Decision Tree) votes on suspicious traffic to confirm if it's a real attack or a false alarm.

Stage 3 (Active Response): Automatically blocks the attacker's Session ID if they cross a strike limit.

📊 Performance Analysis & Results

The model was rigorously tested on a real-world cybersecurity dataset comprising 9,537 network traffic samples.

🏆 Final Model Evaluation

Metric

Score

Insight

Accuracy

92%

High reliability in distinguishing normal vs. attack traffic.

Precision

98%

Extremely low False Positive Rate (Only 2% error in flagging attacks).

Recall

84%

Successfully caught 84% of all real attacks.

Threats Blocked

1,737

Automatically identified and blocked malicious user sessions.

Self-Healing

3,325

False positives identified for future model re-training.

📉 Detailed Classification Report (Hybrid Committee)

              precision    recall  f1-score   support

           0       0.88      0.99      0.93      2960
           1       0.99      0.84      0.91      2376

    accuracy                           0.92      5336


💻 How to Run This Project (Linux/Ubuntu)

Since this project is designed for a Linux environment, follow these steps to run the detection system:

Clone the repository:

git clone [https://github.com/YOUR-USERNAME/Intrusion-Detection-System.git](https://github.com/YOUR-USERNAME/Intrusion-Detection-System.git)
cd Intrusion-Detection-System


Install Dependencies:

pip install pandas numpy scikit-learn torch tqdm


Train the Models:
(This generates the 6 AI models and processors)

python3 train_models.py


Start the Detection Simulation:
(Runs the real-time detection on test data)

python3 detect_final_demo.py


🔮 Future Roadmap (Sem 6)

Real-Time Integration: Connecting the system to live Wi-Fi traffic using Scapy.

Cloud Deployment: Containerizing the app with Docker and deploying on AWS.

Automated Retraining: Implementing a pipeline to auto-update the Gatekeeper model daily.

Created by Manav Bhatt
