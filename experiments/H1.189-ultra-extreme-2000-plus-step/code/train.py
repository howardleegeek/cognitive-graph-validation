"""
H1.189: Attention on 2000+ step ultra-extreme tasks
Extends H1.162 (1500-2000 steps) to even longer sequences.
Tests if attention maintains advantage on 2000-2500 step tasks.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

class UltraExtremeDataset(Dataset):
    def __init__(self, n_samples=100, seq_length=2200, n_features=8):
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
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            seq[:, i*2] = amplitude * np.sin(2 * np.pi * freq * t + phase)
            seq[:, i*2+1] = amplitude * np.cos(2 * np.pi * freq * t + phase)
        return seq
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class ConcatenationModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
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
    def __init__(self, input_dim=8, hidden_dim=128, n_heads=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
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

def train_model(model, train_loader, epochs=10, lr=1e-3):
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
    print("H1.189: Attention on 2000+ step ultra-extreme tasks")
    print("=" * 60)
    
    results = {}
    seq_lengths = [2000, 2200]
    
    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")
        
        train_dataset = UltraExtremeDataset(n_samples=100, seq_length=seq_len)
        test_dataset = UltraExtremeDataset(n_samples=20, seq_length=seq_len)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8)
        
        concat_model = ConcatenationModel(input_dim=8, hidden_dim=128)
        concat_model = train_model(concat_model, train_loader)
        concat_test = evaluate(concat_model, test_loader)
        
        attn_model = AttentionModel(input_dim=8, hidden_dim=128, n_heads=2)
        attn_model = train_model(attn_model, train_loader)
        attn_test = evaluate(attn_model, test_loader)
        
        improvement = (concat_test - attn_test) / concat_test * 100
        
        print(f"Concat Test MSE: {concat_test:.6f}")
        print(f"Attention Test MSE: {attn_test:.6f}")
        print(f"Improvement: {improvement:+.1f}%")
        
        results[seq_len] = {
            "concat_mse": concat_test,
            "attention_mse": attn_test,
            "improvement": improvement
        }
    
    avg_improvement = np.mean([r["improvement"] for r in results.values()])
    print(f"\n=== Average Improvement: {avg_improvement:+.1f}% ===")
    
    status = "SUPPORTED" if avg_improvement > 0 else "REFUTED"
    print(f"Status: {status}")
    
    return results, avg_improvement, status

if __name__ == "__main__":
    results, avg_improvement, status = run_experiment()
    
    output = {
        "hypothesis": "H1.189",
        "statement": "Attention maintains advantage on 2000+ step ultra-extreme multi-step tasks",
        "results": results,
        "average_improvement": avg_improvement,
        "status": status
    }
    
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to results.json")