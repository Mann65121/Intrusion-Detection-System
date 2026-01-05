import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

print("Loading data...")
try:
    df = pd.read_csv('cybersecurity_intrusion_data.csv')
except:
    # Fallback if file not found (creates dummy data for emergency)
    print("CSV not found. Creating dummy data for training...")
    data = {
        'session_id': [f'SID_{i}' for i in range(1000)],
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 1000),
        'packet_size': np.random.randint(50, 1500, 1000),
        'duration': np.random.rand(1000),
        'attack_detected': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)

df.fillna(0, inplace=True) 

# Encoding
encoders = {}
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'attack_detected' in categorical_cols: categorical_cols.remove('attack_detected')

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

labels = df['attack_detected']
features = df.drop('attack_detected', axis=1)
final_columns = features.columns.tolist()

# Scalers
gatekeeper_scaler = MinMaxScaler()
gatekeeper_scaler.fit(features[labels == 0]) # Fit on normal only
investigator_scaler = StandardScaler()
investigator_scaler.fit(features)

# --- GAN Setup ---
features_dim = len(features.columns)
class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)

print("Training Gatekeeper (GAN)...")
gatekeeper = Discriminator(features_dim)
generator = Generator(features_dim)
criterion = nn.BCELoss()
opt_d = optim.Adam(gatekeeper.parameters(), lr=0.0002)
opt_g = optim.Adam(generator.parameters(), lr=0.0002)

normal_data = gatekeeper_scaler.transform(features[labels == 0])
dataset = TensorDataset(torch.tensor(normal_data, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(5): # Short epoch for quick setup
    for (real,) in loader:
        # Train D
        opt_d.zero_grad()
        real_labels = torch.ones(real.size(0), 1)
        fake_labels = torch.zeros(real.size(0), 1)
        loss_d = criterion(gatekeeper(real), real_labels) + criterion(gatekeeper(generator(torch.randn(real.size(0), 10)).detach()), fake_labels)
        loss_d.backward()
        opt_d.step()
        # Train G
        opt_g.zero_grad()
        loss_g = criterion(gatekeeper(generator(torch.randn(real.size(0), 10))), real_labels)
        loss_g.backward()
        opt_g.step()

torch.save(gatekeeper.state_dict(), 'gatekeeper_final.pth')

# --- Investigator Models ---
print("Training Investigator Committee (5 Models)...")
X_scaled = investigator_scaler.transform(features)
y = labels

rf = RandomForestClassifier(n_estimators=10).fit(features, y)
sgd = SGDClassifier(loss='log_loss').fit(X_scaled, y)
knn = KNeighborsClassifier().fit(X_scaled, y)
gnb = GaussianNB().fit(X_scaled, y)
dt = DecisionTreeClassifier().fit(features, y)

# Saving everything
with open('investigator_rf.pkl', 'wb') as f: pickle.dump(rf, f)
with open('investigator_sgd.pkl', 'wb') as f: pickle.dump(sgd, f)
with open('investigator_knn.pkl', 'wb') as f: pickle.dump(knn, f)
with open('investigator_gnb.pkl', 'wb') as f: pickle.dump(gnb, f)
with open('investigator_dt.pkl', 'wb') as f: pickle.dump(dt, f)
with open('scaler_gatekeeper.pkl', 'wb') as f: pickle.dump(gatekeeper_scaler, f)
with open('scaler_investigator.pkl', 'wb') as f: pickle.dump(investigator_scaler, f)
with open('encoders_final.pkl', 'wb') as f: pickle.dump({'encoders': encoders, 'columns': final_columns}, f)

print("ALL MODELS SAVED SUCCESSFULLY!")
