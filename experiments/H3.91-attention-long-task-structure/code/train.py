import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)

class TaskStructuredDataset:
    """Dataset with task structure: goal states, action outcomes"""
    def __init__(self, n_samples: int, seq_len: int, n_objects: int = 2):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_objects = n_objects
        self.state_dim = 4 * n_objects  # position, velocity per object
        self.goal_dim = 4 * n_objects    # goal positions
        self.action_dim = 4              # action parameters
        
    def generate(self):
        X, y = [], []
        for _ in range(self.n_samples):
            # Initial state
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            
            # Add temporal autocorrelation (real robot characteristic)
            for i in range(1, self.seq_len + 1):
                states[i] = 0.7 * states[i-1] + 0.3 * states[i]
            
            # Goal state (task structure!)
            goal = states[-1].copy()
            
            # Action outcomes (task structure!)
            actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
            
            # Combine: states + goal + actions = task structure
            task_structured = np.concatenate([
                states[:-1],           # current states
                np.tile(goal, (self.seq_len, 1)),  # repeated goal
                actions                # action outcomes
            ], axis=1)
            
            X.append(task_structured)
            y.append(states[1:])  # predict next state
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class AttentionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        # Self-attention
        Q = self.query(encoded)
        K = self.key(encoded)
        V = self.value(encoded)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V)
        
        # Use last timestep for prediction
        out = self.output(attended[:, -1])
        return out

class ConcatModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Flatten and predict
        out = self.net(x[:, -1])  # Use last timestep
        return out

def train_model(model, X_train, y_train, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t[:, -1])
        loss.backward()
        optimizer.step()
    
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32)
        pred = model(X_t)
        mse = nn.MSELoss()(pred, y_t[:, -1]).item()
    return mse

def run_experiment():
    results = {
        "experiment_id": "H3.91",
        "hypothesis": "H3.91: Attention on 20+ timesteps WITH task structure",
        "description": "Test attention on longer sequences (20+ timesteps) WITH task structure (goal states, action outcomes)",
        "timestamp": datetime.now().isoformat(),
        "results": []
    }
    
    seq_lengths = [20, 25, 30, 35, 40]
    n_samples = 200
    
    for seq_len in seq_lengths:
        print(f"\n=== Testing seq_len={seq_len} ===")
        
        # Generate data WITH task structure
        dataset = TaskStructuredDataset(n_samples=n_samples, seq_len=seq_len)
        X, y = dataset.generate()
        
        # Split
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        input_dim = X_train.shape[-1]
        
        # Train Concat model
        concat_model = ConcatModel(input_dim)
        concat_model = train_model(concat_model, X_train, y_train)
        concat_mse = evaluate(concat_model, X_test, y_test)
        
        # Train Attention model
        attn_model = AttentionModel(input_dim)
        attn_model = train_model(attn_model, X_train, y_train)
        attn_mse = evaluate(attn_model, X_test, y_test)
        
        delta = ((concat_mse - attn_mse) / concat_mse) * 100
        
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attn MSE: {attn_mse:.6f}")
        print(f"  Delta: {delta:+.1f}%")
        
        results["results"].append({
            "seq_len": seq_len,
            "concat_mse": concat_mse,
            "attn_mse": attn_mse,
            "delta": delta,
            "attn_wins": delta > 0
        })
    
    # Summary
    deltas = [r["delta"] for r in results["results"]]
    avg_delta = np.mean(deltas)
    attn_wins = sum(1 for r in results["results"] if r["attn_wins"])
    
    results["summary"] = {
        "avg_delta": avg_delta,
        "attn_wins": attn_wins,
        "total": len(results["results"]),
        "status": "SUPPORTED" if avg_delta > 0 and attn_wins >= len(results["results"]) * 0.5 else "REFUTED"
    }
    
    print(f"\n=== SUMMARY ===")
    print(f"Average Delta: {avg_delta:+.1f}%")
    print(f"Attention Wins: {attn_wins}/{len(results['results'])}")
    print(f"Status: {results['summary']['status']}")
    
    # Save results
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.91-attention-long-task-structure/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_experiment()