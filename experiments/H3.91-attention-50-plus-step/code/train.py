"""
H3.91: Attention on 50+ step sequences
Builds on H3.34 which showed crossover at 25+ timesteps.
Tests if attention advantage continues at 50+ steps.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

class LongSequenceDataset(Dataset):
    def __init__(self, n_samples=300, seq_length=60, n_features=12):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_features = n_features
        np.random.seed(42)
        self.data = []
        for _ in range(n_samples):
            seq = self._generate_sequence(seq_length, n_features)
            self.data.append(seq)
        self.data = np.array(self.data, dtype=np.float32)
    
    def _generate_sequence(self, length, n_features):
        t = np.linspace(0, 1, length)
        seq = np.zeros((length, n_features))
        for i in range(n_features // 2):
            freq = np.random.uniform(1.0, 3.0)
            phase = np.random.uniform(0, np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            seq[:, i*2] = amplitude * np.sin(2 * np.pi * freq * t + phase)
            seq[:, i*2+1] = amplitude * np.cos(2 * np.pi * freq * t + phase)
        return seq
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class ConcatenationModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        last = encoded[:, -1, :]
        combined = torch.cat([pooled, last], dim=1)
        output = self.decoder(combined)
        return output

class AttentionModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, n_heads=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        encoded = self.encoder(x)
        attended, _ = self.attention(encoded, encoded, encoded)
        output = self.decoder(attended.mean(dim=1))
        return output

class SSMModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, state_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.ssm = nn.Linear(hidden_dim, state_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        state = self.ssm(encoded).mean(dim=1)
        output = self.decoder(state)
        return output

def train_model(model, train_loader, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            target = batch[:, -1, :]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            target = batch[:, -1, :]
            loss = nn.MSELoss()(output, target)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def run_experiment():
    print("=" * 60)
    print("H3.91: Attention on 50+ step sequences")
    print("=" * 60)
    
    results = {}
    seq_lengths = [50, 60, 70, 80, 100]
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_dataset = LongSequenceDataset(n_samples=300, seq_length=seq_len)
        test_dataset = LongSequenceDataset(n_samples=50, seq_length=seq_len)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        concat_model = ConcatenationModel(input_dim=12, hidden_dim=256)
        concat_model = train_model(concat_model, train_loader)
        concat_test = evaluate(concat_model, test_loader)
        
        attn_model = AttentionModel(input_dim=12, hidden_dim=256, n_heads=4)
        attn_model = train_model(attn_model, train_loader)
        attn_test = evaluate(attn_model, test_loader)
        
        ssm_model = SSMModel(input_dim=12, hidden_dim=128, state_dim=16)
        ssm_model = train_model(ssm_model, train_loader)
        ssm_test = evaluate(ssm_model, test_loader)
        
        attn_improvement = (concat_test - attn_test) / concat_test * 100
        ssm_improvement = (concat_test - ssm_test) / concat_test * 100
        
        print(f"Concat MSE: {concat_test:.6f}")
        print(f"Attention MSE: {attn_test:.6f} ({attn_improvement:+.1f}%)")
        print(f"SSM MSE: {ssm_test:.6f} ({ssm_improvement:+.1f}%)")
        
        results[seq_len] = {
            "concat_mse": concat_test,
            "attention_mse": attn_test,
            "ssm_mse": ssm_test,
            "attention_improvement": attn_improvement,
            "ssm_improvement": ssm_improvement
        }
    
    avg_attn = np.mean([r["attention_improvement"] for r in results.values()])
    avg_ssm = np.mean([r["ssm_improvement"] for r in results.values()])
    
    print(f"\n=== Average Attention Improvement: {avg_attn:+.1f}% ===")
    print(f"=== Average SSM Improvement: {avg_ssm:+.1f}% ===")
    
    status = "SUPPORTED" if avg_attn > 0 else "REFUTED"
    print(f"Status: {status}")
    
    return results, avg_attn, avg_ssm, status

if __name__ == "__main__":
    results, avg_attn, avg_ssm, status = run_experiment()
    
    output = {
        "hypothesis": "H3.91",
        "statement": "Attention on 50+ step sequences",
        "results": results,
        "average_attention_improvement": avg_attn,
        "average_ssm_improvement": avg_ssm,
        "status": status
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results.json")