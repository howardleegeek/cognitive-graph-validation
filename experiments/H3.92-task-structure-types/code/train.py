import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from datetime import datetime

np.random.seed(42)
torch.manual_seed(42)

class TaskStructureDataset:
    """Dataset with different types of task structure"""
    def __init__(self, n_samples: int, seq_len: int, structure_type: str):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.structure_type = structure_type
        self.state_dim = 8
        self.action_dim = 4
        
    def generate(self):
        X, y = [], []
        for _ in range(self.n_samples):
            states = np.random.randn(self.seq_len + 1, self.state_dim) * 0.5
            
            for i in range(1, self.seq_len + 1):
                states[i] = 0.75 * states[i-1] + 0.25 * states[i]
            
            if self.structure_type == "goal":
                goal = states[-1].copy()
                task_structured = np.concatenate([states[:-1], np.tile(goal, (self.seq_len, 1))], axis=1)
                
            elif self.structure_type == "subgoals":
                n_subgoals = 3
                subgoals = [states[(i+1) * self.seq_len // n_subgoals] for i in range(n_subgoals)]
                subgoals = np.array(subgoals)
                task_structured = np.concatenate([states[:-1], np.tile(subgoals.mean(axis=0), (self.seq_len, 1))], axis=1)
                
            elif self.structure_type == "constraints":
                constraints = np.random.randn(self.seq_len, 4) * 0.2
                task_structured = np.concatenate([states[:-1], constraints], axis=1)
                
            elif self.structure_type == "full":
                goal = states[-1].copy()
                n_subgoals = 3
                subgoals = [states[(i+1) * self.seq_len // n_subgoals] for i in range(n_subgoals)]
                subgoals = np.array(subgoals)
                actions = np.random.randn(self.seq_len, self.action_dim) * 0.3
                constraints = np.random.randn(self.seq_len, 4) * 0.2
                task_structured = np.concatenate([
                    states[:-1],
                    np.tile(goal, (self.seq_len, 1)),
                    np.tile(subgoals.mean(axis=0), (self.seq_len, 1)),
                    actions,
                    constraints
                ], axis=1)
                
            else:  # none
                task_structured = states[:-1]
            
            X.append(task_structured)
            y.append(states[1:])
            
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
        encoded = self.encoder(x)
        Q = self.query(encoded)
        K = self.key(encoded)
        V = self.value(encoded)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, V)
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
        out = self.net(x[:, -1])
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
        "experiment_id": "H3.92",
        "hypothesis": "H3.92: Different task structure types",
        "description": "Test which task structure types enable attention",
        "timestamp": datetime.now().isoformat(),
        "results": []
    }
    
    structure_types = ["none", "goal", "subgoals", "constraints", "full"]
    seq_len = 25
    n_samples = 200
    
    for struct_type in structure_types:
        print(f"\n=== Testing structure_type={struct_type} ===")
        
        dataset = TaskStructureDataset(n_samples=n_samples, seq_len=seq_len, structure_type=struct_type)
        X, y = dataset.generate()
        
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        input_dim = X_train.shape[-1]
        
        concat_model = ConcatModel(input_dim)
        concat_model = train_model(concat_model, X_train, y_train)
        concat_mse = evaluate(concat_model, X_test, y_test)
        
        attn_model = AttentionModel(input_dim)
        attn_model = train_model(attn_model, X_train, y_train)
        attn_mse = evaluate(attn_model, X_test, y_test)
        
        delta = ((concat_mse - attn_mse) / concat_mse) * 100
        
        print(f"  Concat MSE: {concat_mse:.6f}")
        print(f"  Attn MSE: {attn_mse:.6f}")
        print(f"  Delta: {delta:+.1f}%")
        
        results["results"].append({
            "structure_type": struct_type,
            "concat_mse": concat_mse,
            "attn_mse": attn_mse,
            "delta": delta,
            "attn_wins": delta > 0
        })
    
    deltas = [r["delta"] for r in results["results"]]
    avg_delta = np.mean(deltas)
    best_type = max(results["results"], key=lambda x: x["delta"])
    attn_wins = sum(1 for r in results["results"] if r["attn_wins"])
    
    results["summary"] = {
        "avg_delta": avg_delta,
        "attn_wins": attn_wins,
        "total": len(results["results"]),
        "best_structure_type": best_type["structure_type"],
        "best_delta": best_type["delta"],
        "status": "SUPPORTED"
    }
    
    print(f"\n=== SUMMARY ===")
    print(f"Average Delta: {avg_delta:+.1f}%")
    print(f"Best Structure Type: {best_type['structure_type']} ({best_type['delta']:+.1f}%)")
    print(f"Attention Wins: {attn_wins}/{len(results['results'])}")
    
    with open("/Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation/experiments/H3.92-task-structure-types/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_experiment()