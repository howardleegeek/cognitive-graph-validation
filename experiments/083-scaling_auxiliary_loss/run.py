#!/usr/bin/env python3
"""
H1.470.1.1.36: Scaling Auxiliary Loss Benefits with Model Size and Data Volume

Context:
- H1.470.1.1.34: Auxiliary losses SUPPORTED (temporal consistency +5.70% on multi-step)
- H1.470.1.1.35: Experience replay INCONCLUSIVE (best +0.49%, replay adds noise)
- Recommendation: Test whether auxiliary loss benefits scale with model size and data volume

Hypothesis: Auxiliary loss benefits (particularly temporal consistency) will scale
positively with both model size (more parameters = better regularization benefit)
and data volume (more data = more stable auxiliary signal).

Predictions:
1. Larger models will show greater benefit from temporal consistency loss
2. More training data will amplify auxiliary loss benefits
3. The interaction effect (large model + more data + auxiliary loss) will be superlinear

Configurations:
- Model sizes: small (32 hidden), medium (64 hidden), large (128 hidden)
- Data volumes: 500, 1000, 2000 samples
- Loss types: baseline MSE vs temporal consistency
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ACTION_DIM = 7
N_EPOCHS = 40
BATCH_SIZE = 64
TEMPORAL_WEIGHT = 0.3


class CognitiveGraphModel(nn.Module):
    def __init__(self, obs_dim=128, action_dim=ACTION_DIM, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.physical_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        # Unified representation: physical (hidden//2) + semantic (hidden*2) = hidden*2.5
        unified_dim = hidden_dim // 2 + hidden_dim * 2
        self.graph_processor = nn.GRU(unified_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        if len(unified.shape) == 2:
            unified = unified.unsqueeze(1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out[:, -1, :])
    
    def get_sequence_output(self, x):
        """Get output for entire sequence for temporal consistency loss."""
        physical = self.physical_encoder(x)
        semantic = self.semantic_encoder(x)
        unified = torch.cat([physical, semantic], dim=-1)
        if len(unified.shape) == 2:
            unified = unified.unsqueeze(1)
        out, _ = self.graph_processor(unified)
        return self.decoder(out.squeeze(1))  # [batch, seq_len, action_dim]


def generate_multi_step_data(n_samples, obs_dim=128, max_steps=4):
    """Generate multi-step manipulation data with temporal structure."""
    data = []
    for _ in range(n_samples):
        n_steps = random.randint(2, max_steps)
        trajectory = []
        
        # Initial state
        state = np.random.randn(obs_dim).astype(np.float32) * 0.5
        
        for step in range(n_steps):
            # Action depends on state and step
            action = np.tanh(state[:7] + 0.3 * np.sin(step * 0.5)) * 0.8
            action += np.random.randn(7).astype(np.float32) * 0.05  # Noise
            
            # State transition
            next_state = state + np.random.randn(obs_dim).astype(np.float32) * 0.1
            next_state[:7] = next_state[:7] * 0.9 + action * 0.1
            
            trajectory.append({
                'state': state.copy(),
                'action': action.copy(),
                'next_state': next_state.copy()
            })
            state = next_state
        
        data.append(trajectory)
    
    return data


def temporal_consistency_loss(model, batch_states, batch_actions, pred_actions):
    """
    Compute temporal consistency loss: predictions should be smooth across timesteps.
    """
    if len(batch_states.shape) < 3:
        return torch.tensor(0.0)
    
    # Get predictions for full sequence
    seq_preds = model.get_sequence_output(batch_states)  # [batch, seq, action]
    
    # Temporal smoothness: consecutive predictions should be similar
    if seq_preds.shape[1] > 1:
        diff = seq_preds[:, 1:, :] - seq_preds[:, :-1, :]
        tc_loss = torch.mean(diff ** 2)
    else:
        tc_loss = torch.tensor(0.0)
    
    return tc_loss


def train_model(model, train_data, val_data, use_temporal_loss=False, epochs=N_EPOCHS):
    """Train model with optional temporal consistency loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle training data
        random.shuffle(train_data)
        
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            
            # Prepare batch
            states = torch.tensor(np.array([t['state'] for t in batch]), dtype=torch.float32)
            actions = torch.tensor(np.array([t['action'] for t in batch]), dtype=torch.float32)
            
            optimizer.zero_grad()
            
            pred_actions = model(states)
            
            # MSE loss
            mse_loss = F.mse_loss(pred_actions, actions)
            
            # Temporal consistency loss
            if use_temporal_loss:
                tc_loss = temporal_consistency_loss(model, states, actions, pred_actions)
                total_loss = mse_loss + TEMPORAL_WEIGHT * tc_loss
            else:
                total_loss = mse_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += mse_loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_states = torch.tensor(np.array([t['state'] for t in val_data]), dtype=torch.float32)
            val_actions = torch.tensor(np.array([t['action'] for t in val_data]), dtype=torch.float32)
            val_pred = model(val_states)
            val_loss = F.mse_loss(val_pred, val_actions).item()
            val_losses.append(val_loss)
    
    return train_losses, val_losses


def run_experiment():
    """Run the scaling experiment."""
    results = {
        "experiment_id": "H1.470.1.1.36",
        "description": "Scaling Auxiliary Loss Benefits with Model Size and Data Volume",
        "timestamp": datetime.now().isoformat(),
        "configurations": [],
        "key_findings": {}
    }
    
    # Configuration matrix
    hidden_dims = [32, 64, 128]  # small, medium, large
    data_volumes = [500, 1000, 2000]  # samples
    loss_types = ["baseline", "temporal_consistency"]
    
    obs_dim = 128
    
    all_results = {}
    
    for hidden_dim in hidden_dims:
        for n_samples in data_volumes:
            for loss_type in loss_types:
                config_name = f"h{hidden_dim}_n{n_samples}_{loss_type}"
                print(f"\nRunning: {config_name}")
                
                # Generate data
                train_data_raw = generate_multi_step_data(n_samples, obs_dim=obs_dim)
                val_data_raw = generate_multi_step_data(n_samples // 5, obs_dim=obs_dim)
                
                # Flatten for training (use first step of each trajectory)
                train_data = []
                for traj in train_data_raw:
                    train_data.extend(traj[:2])  # Use first 2 steps
                
                val_data = []
                for traj in val_data_raw:
                    val_data.extend(traj[:2])
                
                # Create model
                model = CognitiveGraphModel(obs_dim=obs_dim, hidden_dim=hidden_dim)
                
                # Train
                use_tc = loss_type == "temporal_consistency"
                train_losses, val_losses = train_model(
                    model, train_data, val_data,
                    use_temporal_loss=use_tc,
                    epochs=N_EPOCHS
                )
                
                final_val_loss = val_losses[-1]
                
                config_result = {
                    "hidden_dim": hidden_dim,
                    "n_samples": n_samples,
                    "loss_type": loss_type,
                    "final_train_loss": train_losses[-1],
                    "final_val_loss": final_val_loss,
                    "min_val_loss": min(val_losses)
                }
                
                results["configurations"].append(config_result)
                all_results[config_name] = final_val_loss
                
                print(f"  Val loss: {final_val_loss:.6f}")
    
    # Analyze scaling effects
    print("\n=== Analyzing scaling effects ===")
    
    # Effect of model size (at fixed data volume)
    for n_samples in data_volumes:
        print(f"\nData volume: {n_samples} samples")
        for hidden_dim in hidden_dims:
            baseline_key = f"h{hidden_dim}_n{n_samples}_baseline"
            tc_key = f"h{hidden_dim}_n{n_samples}_temporal_consistency"
            
            if baseline_key in all_results and tc_key in all_results:
                baseline_loss = all_results[baseline_key]
                tc_loss = all_results[tc_key]
                improvement = (baseline_loss - tc_loss) / baseline_loss * 100
                
                print(f"  Hidden {hidden_dim}: baseline={baseline_loss:.6f}, TC={tc_loss:.6f}, improvement={improvement:+.2f}%")
    
    # Effect of data volume (at fixed model size)
    for hidden_dim in hidden_dims:
        print(f"\nModel size: hidden_dim={hidden_dim}")
        for n_samples in data_volumes:
            baseline_key = f"h{hidden_dim}_n{n_samples}_baseline"
            tc_key = f"h{hidden_dim}_n{n_samples}_temporal_consistency"
            
            if baseline_key in all_results and tc_key in all_results:
                baseline_loss = all_results[baseline_key]
                tc_loss = all_results[tc_key]
                improvement = (baseline_loss - tc_loss) / baseline_loss * 100
                
                print(f"  N={n_samples}: baseline={baseline_loss:.6f}, TC={tc_loss:.6f}, improvement={improvement:+.2f}%")
    
    # Compute key metrics
    improvements_by_model = {h: [] for h in hidden_dims}
    improvements_by_data = {n: [] for n in data_volumes}
    
    for hidden_dim in hidden_dims:
        for n_samples in data_volumes:
            baseline_key = f"h{hidden_dim}_n{n_samples}_baseline"
            tc_key = f"h{hidden_dim}_n{n_samples}_temporal_consistency"
            
            if baseline_key in all_results and tc_key in all_results:
                improvement = (all_results[baseline_key] - all_results[tc_key]) / all_results[baseline_key] * 100
                improvements_by_model[hidden_dim].append(improvement)
                improvements_by_data[n_samples].append(improvement)
    
    # Average improvement by model size
    avg_by_model = {h: np.mean(v) for h, v in improvements_by_model.items() if v}
    avg_by_data = {n: np.mean(v) for n, v in improvements_by_data.items() if v}
    
    results["key_findings"] = {
        "avg_improvement_by_model_size": {str(k): f"{v:+.2f}%" for k, v in avg_by_model.items()},
        "avg_improvement_by_data_volume": {str(k): f"{v:+.2f}%" for k, v in avg_by_data.items()},
        "best_configuration": min(all_results.items(), key=lambda x: x[1])[0],
        "best_loss": min(all_results.values())
    }
    
    # Determine conclusion
    # Check if larger models benefit more from TC
    model_sizes = sorted(avg_by_model.keys())
    if len(model_sizes) >= 2:
        smaller_improvement = avg_by_model[model_sizes[0]]
        larger_improvement = avg_by_model[model_sizes[-1]]
        model_scaling = larger_improvement > smaller_improvement
    else:
        model_scaling = False
    
    # Check if more data benefits more from TC
    data_sizes = sorted(avg_by_data.keys())
    if len(data_sizes) >= 2:
        less_data_improvement = avg_by_data[data_sizes[0]]
        more_data_improvement = avg_by_data[data_sizes[-1]]
        data_scaling = more_data_improvement > less_data_improvement
    else:
        data_scaling = False
    
    if model_scaling and data_scaling:
        conclusion = "SUPPORTED"
        conclusion_detail = "Auxiliary loss benefits scale positively with both model size and data volume"
    elif model_scaling:
        conclusion = "PARTIALLY_SUPPORTED"
        conclusion_detail = "Auxiliary loss benefits scale with model size but not data volume"
    elif data_scaling:
        conclusion = "PARTIALLY_SUPPORTED"
        conclusion_detail = "Auxiliary loss benefits scale with data volume but not model size"
    else:
        conclusion = "REFUTED"
        conclusion_detail = "Auxiliary loss benefits do not scale with model size or data volume"
    
    results["conclusion"] = conclusion
    results["conclusion_detail"] = conclusion_detail
    results["model_scaling"] = model_scaling
    results["data_scaling"] = data_scaling
    
    # Save results
    output_dir = Path(__file__).parent
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== CONCLUSION: {conclusion} ===")
    print(f"Detail: {conclusion_detail}")
    print(f"Model scaling: {model_scaling}")
    print(f"Data scaling: {data_scaling}")
    print(f"\nResults saved to {output_dir / 'results.json'}")
    
    return results


if __name__ == "__main__":
    run_experiment()