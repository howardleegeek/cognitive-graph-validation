"""
H1.109: Complex Compositional Multi-Step Tasks
Tests unified cognitive graph architecture on 20-40 step tasks with multiple compositional subtasks
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import json

np.random.seed(42)
torch.manual_seed(42)

class ComplexCompositionalDataset:
    """Generate complex compositional multi-step tasks"""
    
    def __init__(self, n_samples: int = 200, n_steps: int = 30, n_subtasks: int = 4):
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_subtasks = n_subtasks
        
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate compositional task data"""
        X, y = [], []
        
        for _ in range(self.n_samples):
            # Generate task with compositional structure
            # Each subtask has 5-8 steps
            steps_per_subtask = self.n_steps // self.n_subtasks
            
            task_X = []
            task_y = []
            
            for subtask in range(self.n_subtasks):
                # Each subtask has different dynamics
                base_friction = 0.1 + subtask * 0.05
                base_mass = 1.0 + subtask * 0.2
                
                for step in range(steps_per_subtask):
                    # State: position + velocity + object features + subtask encoding
                    state = np.random.randn(12)
                    state[0] = step / steps_per_subtask  # Progress within subtask
                    state[1] = subtask / self.n_subtasks  # Subtask ID
                    state[2] = base_friction + np.random.randn() * 0.02
                    state[3] = base_mass + np.random.randn() * 0.1
                    
                    # Action: 4D
                    action = np.random.randn(4)
                    
                    # Next state depends on subtask dynamics
                    next_state = state.copy()
                    next_state[:4] += action[:4] * 0.1
                    next_state[4:8] += np.random.randn(4) * 0.05
                    
                    task_X.append(np.concatenate([state, action]))
                    task_y.append(next_state[:8])
            
            X.append(np.array(task_X))
            y.append(np.array(task_y))
        
        return np.array(X), np.array(y)


class BaselineModel(nn.Module):
    """Standard MLP concatenation baseline"""
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class UnifiedModel(nn.Module):
    """Unified cognitive graph architecture"""
    def __init__(self, input_dim: int = 16, unified_dim: int = 256, output_dim: int = 8):
        super().__init__()
        # Physical branch (22% = 56 dims)
        self.physical_encoder = nn.Linear(12, 56)
        # Semantic branch (78% = 200 dims)
        self.semantic_encoder = nn.Linear(4, 200)
        # Unified processing
        self.unified_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Output
        self.output_net = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # Split input into state and action
        state = x[:, :12]
        action = x[:, 12:]
        
        # Encode physical (state) and semantic (action)
        physical = self.physical_encoder(state)
        semantic = self.semantic_encoder(action)
        
        # Unified representation
        unified = torch.cat([physical, semantic], dim=-1)
        processed = self.unified_net(unified)
        
        return self.output_net(processed)


class UnifiedAttnModel(nn.Module):
    """Unified with cross-modal attention"""
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128, output_dim: int = 8):
        super().__init__()
        # Encoders
        self.state_encoder = nn.Linear(12, hidden_dim)
        self.action_encoder = nn.Linear(4, hidden_dim)
        
        # Attention
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        state = x[:, :12]
        action = x[:, 12:]
        
        state_enc = self.state_encoder(state).unsqueeze(1)
        action_enc = self.action_encoder(action).unsqueeze(1)
        
        # Cross-attention
        attn_out, _ = self.attn(action_enc, state_enc, state_enc)
        attn_out = attn_out.squeeze(1)
        
        combined = torch.cat([state_enc.squeeze(1), attn_out], dim=-1)
        return self.output_net(combined)


class UnifiedSSMModel(nn.Module):
    """Unified with SSM (from H1.102)"""
    def __init__(self, input_dim: int = 16, ssm_state: int = 16, hidden: int = 256, output_dim: int = 8):
        super().__init__()
        self.output_dim = output_dim
        # Encoders
        self.state_encoder = nn.Linear(12, 128)
        self.action_encoder = nn.Linear(4, 128)
        
        # SSM state space - B maps from combined (256) to ssm_state
        self.ssm_B = nn.Linear(256, ssm_state)
        self.ssm_C = nn.Linear(ssm_state, output_dim)
        
        # Hidden processing
        self.hidden_net = nn.Sequential(
            nn.Linear(256 + ssm_state, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
    def forward(self, x):
        state = x[:, :12]
        action = x[:, 12:]
        
        state_enc = self.state_encoder(state)
        action_enc = self.action_encoder(action)
        
        combined = torch.cat([state_enc, action_enc], dim=-1)
        
        # SSM dynamics
        ssm_state = torch.tanh(self.ssm_B(combined))
        ssm_out = self.ssm_C(ssm_state)
        
        # Combine with hidden
        hidden_combined = torch.cat([combined, ssm_state], dim=-1)
        hidden = self.hidden_net(hidden_combined)
        
        return hidden[:, :self.output_dim] + ssm_out


def train_model(model, X_train, y_train, epochs: int = 100):
    """Train model and return final loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_loss = criterion(model(X_tensor), y_tensor).item()
    
    return final_loss


def evaluate(model, X_test, y_test):
    """Evaluate model on test data"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.FloatTensor(y_test)
        pred = model(X_tensor)
        mse = nn.MSELoss()(pred, y_tensor).item()
    return mse


def run_experiment():
    """Run H1.109 experiment"""
    print("=" * 60)
    print("H1.109: Complex Compositional Multi-Step Tasks")
    print("=" * 60)
    
    results = {}
    
    # Test different task complexities
    for n_steps in [20, 30, 40]:
        print(f"\n--- Testing {n_steps}-step tasks ---")
        
        # Generate data
        dataset = ComplexCompositionalDataset(n_samples=200, n_steps=n_steps, n_subtasks=4)
        X, y = dataset.generate()
        
        # Flatten for training
        X_flat = X.reshape(-1, 16)
        y_flat = y.reshape(-1, 8)
        
        # Split
        n_train = int(0.8 * len(X_flat))
        X_train, X_test = X_flat[:n_train], X_flat[n_train:]
        y_train, y_test = y_flat[:n_train], y_flat[n_train:]
        
        # Train and evaluate each model
        models = {
            'Baseline': BaselineModel(16, 128, 8),
            'Unified': UnifiedModel(16, 256, 8),
            'Unified+Attn': UnifiedAttnModel(16, 128, 8),
            'Unified+SSM': UnifiedSSMModel(16, 16, 256, 8),
        }
        
        step_results = {}
        for name, model in models.items():
            train_model(model, X_train, y_train)
            mse = evaluate(model, X_test, y_test)
            step_results[name] = mse
            print(f"  {name}: MSE = {mse:.6f}")
        
        results[f"{n_steps}_step"] = step_results
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    improvements = {}
    for n_steps in [20, 30, 40]:
        baseline = results[f"{n_steps}_step"]['Baseline']
        for model in ['Unified', 'Unified+Attn', 'Unified+SSM']:
            mse = results[f"{n_steps}_step"][model]
            improvement = (baseline - mse) / baseline * 100
            if model not in improvements:
                improvements[model] = []
            improvements[model].append(improvement)
            print(f"{n_steps}-step {model}: {improvement:+.1f}%")
    
    # Average improvements
    print("\n--- Average Improvement ---")
    for model, imps in improvements.items():
        avg = np.mean(imps)
        print(f"{model}: {avg:+.1f}%")
    
    # Determine status
    best_model = max(improvements, key=lambda k: np.mean(improvements[k]))
    best_avg = np.mean(improvements[best_model])
    
    status = "SUPPORTED" if best_avg > 5 else ("MARGINAL" if best_avg > 0 else "REFUTED")
    
    print(f"\nStatus: {status} ({best_model}: {best_avg:+.1f}%)")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump({
            'results': results,
            'improvements': improvements,
            'status': status,
            'best_model': best_model,
            'best_avg': best_avg
        }, f, indent=2)
    
    return results, status, best_model, best_avg


if __name__ == "__main__":
    results, status, best_model, best_avg = run_experiment()