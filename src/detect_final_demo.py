import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import time
import warnings
from collections import Counter 
from sklearn.metrics import classification_report

# 1. Filter warnings
warnings.filterwarnings("ignore")

# --- GAN Class Definition ---
# (Must match the class structure used during training)
class Discriminator(nn.Module):
    def __init__(self, features_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- Configuration ---
TEST_FILE = 'cybersecurity_intrusion_data.csv'
ANOMALY_THR = 0.8
STRIKE_LIMIT = 5
VERBOSE = True # Set to False to speed up processing (hides individual packet logs)

print("\n" + "="*50)
print("   HYBRID AI INTRUSION PREVENTION SYSTEM (IPS)")
print("="*50)

# --- Load Resources ---
try:
    print("Loading Models and Encoders...")
    with open('encoders_final.pkl', 'rb') as f: data = pickle.load(f)
    encoders, cols = data['encoders'], data['columns']
    
    with open('scaler_gatekeeper.pkl', 'rb') as f: sc_gk = pickle.load(f)
    with open('scaler_investigator.pkl', 'rb') as f: sc_inv = pickle.load(f)
    
    models = {}
    model_files = {
        'RF': 'investigator_rf.pkl',
        'SGD': 'investigator_sgd.pkl',
        'KNN': 'investigator_knn.pkl',
        'GNB': 'investigator_gnb.pkl',
        'DT': 'investigator_dt.pkl'
    }
    
    for name, filename in model_files.items():
        with open(filename, 'rb') as f: models[name] = pickle.load(f)
    
    gatekeeper = Discriminator(len(cols))
    gatekeeper.load_state_dict(torch.load('gatekeeper_final.pth'))
    gatekeeper.eval()
    print("All Systems Online.")

except FileNotFoundError as e:
    print(f"\n[!] ERROR: Missing file - {e}")
    print("Please run the training script first.")
    exit()

# --- Load Data ---
try:
    df = pd.read_csv(TEST_FILE).fillna(0)
except FileNotFoundError:
    print(f"[!] ERROR: {TEST_FILE} not found.")
    exit()

if 'session_id' not in df.columns:
    df['session_id'] = [f"SID_{i}" for i in range(len(df))]

# --- Setup Trackers ---
results = {
    'true': [],
    'committee': [],
    'RF': [], 'SGD': [], 'KNN': [], 'GNB': [], 'DT': []
}
user_strikes = {}
blocked_users = set()

# --- Pre-processing ---
print("Pre-processing data for simulation...")
X_raw = df.drop(['attack_detected'], axis=1, errors='ignore')

# Apply Label Encoding
for c in X_raw.columns:
    if c in encoders:
        # Handle unseen labels safely
        X_raw[c] = X_raw[c].astype(str).apply(lambda x: encoders[c].transform([x])[0] if x in encoders[c].classes_ else 0)

# Ensure column order matches training
X_ordered = X_raw[cols]

# Scale features
X_gk_scaled = torch.tensor(sc_gk.transform(X_ordered), dtype=torch.float32)
X_inv_scaled = sc_inv.transform(X_ordered)

print("\n--- STARTING LIVE DETECTION STREAM ---")
print("(Press Ctrl+C to stop simulation and generate report immediately)")
time.sleep(1)

RED, GREEN, YEL, RES = '\033[91m', '\033[92m', '\033[93m', '\033[0m'

# --- MAIN LOOP WITH INTERRUPT HANDLING ---
try:
    for i in range(len(df)):
        user = str(df.iloc[i]['session_id'])
        true_label = str(df.iloc[i]['attack_detected'])
        
        # Check Blocklist
        if user in blocked_users:
            results['committee'].append('Blocked')
            results['true'].append(true_label)
            # Fill individual models with placeholders to keep list lengths synced
            for m in ['RF', 'SGD', 'KNN', 'GNB', 'DT']:
                results[m].append('Blocked')
            continue
        
        # 1. Stage 1: Gatekeeper (GAN)
        with torch.no_grad():
            score = gatekeeper(X_gk_scaled[i].unsqueeze(0)).item()
        
        # 2. Stage 2: Investigator Committee
        # We process single row slices to simulate real-time arrival
        curr_X_ord = X_ordered.iloc[i:i+1].values
        curr_X_inv = X_inv_scaled[i:i+1].reshape(1, -1)
        
        p_rf  = str(models['RF'].predict(curr_X_ord)[0])
        p_sgd = str(models['SGD'].predict(curr_X_inv)[0])
        p_knn = str(models['KNN'].predict(curr_X_inv)[0])
        p_gnb = str(models['GNB'].predict(curr_X_inv)[0])
        p_dt  = str(models['DT'].predict(curr_X_ord)[0])

        # Hybrid Decision Logic
        if score > ANOMALY_THR:
            final_pred = '0' # Gatekeeper sees Normal
        else:
            # Suspicious -> Committee Votes
            votes = [p_rf, p_sgd, p_knn, p_gnb, p_dt]
            final_pred = Counter(votes).most_common(1)[0][0]
            
            # IPS Active Response
            attack_votes = votes.count('1')
            if final_pred == '1':
                strikes = 1 if attack_votes <= 3 else 5 # Aggressive blocking if unanimous
                user_strikes[user] = user_strikes.get(user, 0) + strikes
                
                if VERBOSE:
                    print(f"{RED}>>> ALERT! User: {user} | Score: {score:.2f} | Committee: Attack ({attack_votes}/5){RES}")
                
                if user_strikes[user] >= STRIKE_LIMIT:
                    print(f"{RED}>>> ACTION: BLOCKING SESSION {user} (Strike Limit Exceeded){RES}")
                    blocked_users.add(user)
            else:
                if VERBOSE:
                    print(f"{YEL}>>> SUSPICIOUS | Score: {score:.2f} | Committee: Normal | {GREEN}False Positive Filtered{RES}")

        # Log results
        results['true'].append(true_label)
        results['committee'].append(final_pred)
        results['RF'].append(p_rf)
        results['SGD'].append(p_sgd)
        results['KNN'].append(p_knn)
        results['GNB'].append(p_gnb)
        results['DT'].append(p_dt)

except KeyboardInterrupt:
    print(f"\n{YEL}[!] Simulation Interrupted by User.{RES}")
    print(f"{YEL}[!] Generating Report for processed data ({len(results['true'])} records)...{RES}")

# --- REPORT GENERATION ---
def print_report(name, y_true, y_pred):
    # Align lengths just in case interrupt happened mid-append
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Filter out 'Blocked' for classification metrics (as they are technically pre-empted)
    clean_data = [(t, p) for t, p in zip(y_true, y_pred) if p != 'Blocked']
    
    if not clean_data:
        print(f"\n[REPORT FOR: {name}] - No data processed or all users blocked.")
        return

    clean_true, clean_pred = zip(*clean_data)
    
    print(f"\n[REPORT FOR: {name}]")
    print(classification_report(clean_true, clean_pred, zero_division=0))

print("\n" + "="*60)
print("          DETAILED PERFORMANCE ANALYSIS REPORT")
print("="*60)

print_report("RANDOM FOREST", results['true'], results['RF'])
print_report("SGD CLASSIFIER", results['true'], results['SGD'])
print_report("K-NEAREST NEIGHBORS", results['true'], results['KNN'])
print_report("GAUSSIAN NAIVE BAYES", results['true'], results['GNB'])
print_report("DECISION TREE", results['true'], results['DT'])

print("\n" + "="*60)
print("         FINAL HYBRID COMMITTEE (ENSEMBLE) RESULTS")
print("="*60)
print_report("HYBRID IPS COMMITTEE", results['true'], results['committee'])

print(f"\nSummary:")
print(f"Total Processed: {len(results['true'])}")
print(f"Total Blocked Users: {len(blocked_users)}")
