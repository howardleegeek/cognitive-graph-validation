"""
H1.196: Complex Multi-Step Deepening - Testing attention on 20-40 step complex tasks 
with autocorrelation injection, building on H1.181's success showing +26.9% at ρ=0.95.

Hypothesis: Attention advantage continues to grow on complex multi-step tasks (>20 steps) 
when temporal structure (autocorrelation) is preserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

class UnifiedAttentionModel(nn.Module):
    """Unified architecture with attention for complex multi-step tasks."""
    
    def __init__(self, state_dim=16, action_dim=8, hidden_dim=512, num_steps=20):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Unified encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention for temporal reasoning
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            dropout=0.1
        )
        
        # Action-conditioned gating
        self.action_gate = nn.Linear(action_dim, hidden_dim)
        
        # Output predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, states, actions, return_attention_weights=False):
        batch_size = states.shape[0]
        
        # Encode state-action pairs
        pairs = torch.cat([states, actions], dim=-1)
        encoded = self.encoder(pairs)
        
        # Transpose for attention: (num_steps, batch, hidden_dim)
        encoded_trans = encoded.transpose(0, 1)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            encoded_trans, encoded_trans, encoded_trans
        )
        
        # Take the last timestep output: (batch, hidden_dim)
        last_output = attn_output[-1]
        
        # Predict next action
        pred = self.predictor(last_output)
        
        if return_attention_weights:
            return pred, attn_weights
        return pred


class BaselineModel(nn.Module):
    """Baseline concatenation model."""
    
    def __init__(self, state_dim=16, action_dim=8, hidden_dim=512, num_steps=20):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple concatenation + MLP
        self.network = nn.Sequential(
            nn.Linear(state_dim * num_steps + action_dim * num_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, states, actions):
        batch_size = states.shape[0]
        
        # Flatten
        states_flat = states.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        
        # Concatenate
        combined = torch.cat([states_flat, actions_flat], dim=-1)
        return self.network(combined)


def generate_complex_trajectory(num_steps, state_dim, action_dim, autocorr=0.85, noise=0.01):
    """Generate complex multi-step trajectory with autocorrelation."""
    states = []
    actions = []
    targets = []  # Next action to predict
    
    # Initial state
    state = np.random.randn(state_dim) * 0.1
    
    for t in range(num_steps):
        # Previous action influences current state via autocorrelation
        if len(actions) > 0:
            prev_action = actions[-1]
            # Project action to state dim for dot product
            prev_action_padded = np.concatenate([prev_action, np.zeros(state_dim - len(prev_action))])
            # Autocorrelation: current state influenced by previous action
            state = autocorr * np.dot(prev_action_padded, state) + np.random.randn(state_dim) * noise
        else:
            state = np.random.randn(state_dim) * 0.1
        
        # Action is function of state + temporal structure
        action = np.tanh(state[:action_dim] * 0.5 + 0.1 * np.sin(np.arange(action_dim) * 0.5))
        
        # Next action (target) is next step's action, with some dynamics
        next_action = np.tanh(state[:action_dim] * 0.5 + 0.15 * np.sin(np.arange(action_dim) * 0.5 + 0.5))
        
        states.append(state.copy())
        actions.append(action.copy())
        targets.append(next_action.copy())
        
        # Update state for next iteration
        action_padded = np.concatenate([action, np.zeros(state_dim - action_dim)])
        state = state + 0.1 * action_padded + np.random.randn(state_dim) * noise
    
    return (np.array(states), np.array(actions), np.array(targets))


def train_model(model, train_loader, epochs=100, lr=1e-3):
    """Train model on complex multi-step tasks."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for states, actions, targets in train_loader:
            optimizer.zero_grad()
            
            preds = model(states, actions)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
    
    return losses


def evaluate_model(model, test_loader):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for states, actions, targets in test_loader:
            preds = model(states, actions)
            loss = F.mse_loss(preds, targets)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def run_experiment():
    """Run H1.196: Complex multi-step deepening."""
    
    print("=" * 60)
    print("H1.196: Complex Multi-Step Deepening")
    print("Testing attention on 20-40 step complex tasks")
    print("Building on H1.181: +26.9% at ρ=0.95")
    print("=" * 60)
    
    # Configuration - reduced for speed
    state_dim = 16
    action_dim = 8
    hidden_dim = 256  # Reduced from 512
    num_samples = 200  # Reduced from 500
    num_epochs = 50    # Reduced from 100
    autocorr_levels = [0.7, 0.95]  # Reduced from 3
    step_levels = [20, 40]  # Reduced from 3
    
    results = []
    
    for autocorr in autocorr_levels:
        for num_steps in step_levels:
            print(f"\n--- Autocorr={autocorr}, Steps={num_steps} ---")
            
            # Generate data
            train_states, train_actions, train_targets_arr = [], [], []
            for _ in range(num_samples):
                states, actions, targets = generate_complex_trajectory(
                    num_steps, state_dim, action_dim, autocorr=autocorr
                )
                train_states.append(states)
                train_actions.append(actions)
                train_targets_arr.append(targets[-1])  # Final action as target
            
            train_states = torch.FloatTensor(np.array(train_states))
            train_actions = torch.FloatTensor(np.array(train_actions))
            train_targets = torch.FloatTensor(np.array(train_targets_arr))
            
            train_dataset = TensorDataset(train_states, train_actions, train_targets)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Train baseline
            baseline = BaselineModel(state_dim, action_dim, hidden_dim, num_steps)
            baseline_losses = train_model(baseline, train_loader, epochs=100)
            baseline_loss = evaluate_model(baseline, train_loader)
            
            # Train attention model
            attention = UnifiedAttentionModel(state_dim, action_dim, hidden_dim, num_steps)
            attention_losses = train_model(attention, train_loader, epochs=100)
            attention_loss = evaluate_model(attention, train_loader)
            
            # Calculate improvement
            improvement = (baseline_loss - attention_loss) / baseline_loss * 100
            
            results.append({
                'autocorr': autocorr,
                'num_steps': num_steps,
                'baseline_mse': baseline_loss,
                'attention_mse': attention_loss,
                'improvement': improvement
            })
            
            print(f"  Baseline MSE: {baseline_loss:.6f}")
            print(f"  Attention MSE: {attention_loss:.6f}")
            print(f"  Improvement: {improvement:+.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: H1.196 Complex Multi-Step Deepening")
    print("=" * 60)
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    print(f"Average improvement: {avg_improvement:+.1f}%")
    
    # Best configuration
    best = max(results, key=lambda x: x['improvement'])
    print(f"Best: autocorr={best['autocorr']}, steps={best['num_steps']} → {best['improvement']:+.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_experiment()