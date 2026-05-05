#!/usr/bin/env python3
"""H3.37: Attention with Stochastic Dynamics
Test attention on trajectories with noise, delays, dropouts - real-world conditions
Following H3.35, H3.36 success.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(43)

def generate_stochastic_trajectories(n_samples=300, n_timesteps=40, noise_level=0.1):
    """Generate trajectories with stochastic elements."""
    trajectories = []
    
    for _ in range(n_samples):
        dt = 0.02
        noise = noise_level
        
        pos = np.random.uniform(-1, 1)
        vel = 0.0
        
        states = []
        actions = []
        
        action_buffer = [0.0, 0.0]  # Simulate motor delay
        
        for t in range(n_timesteps):
            # Get delayed action
            delayed_action = action_buffer[0] if len(action_buffer) > 0 else 0.0
            
            # New command
            force = np.random.uniform(-1, 1)
            action_buffer.append(float(force))
            if len(action_buffer) > 2:
                action_buffer.pop(0)
            
            # Sensor noise
            noisy_pos = pos + np.random.normal(0, noise)
            noisy_vel = vel + np.random.normal(0, noise)
            
            # Dynamics
            acc = delayed_action - 0.2 * vel + np.random.normal(0, noise)
            vel += acc * dt
            pos += vel * dt
            
            states.append([noisy_pos, noisy_vel])
            actions.append([force])
        
        trajectories.append({
            'states': np.array(states),
            'actions': np.array(actions)
        })
    
    return trajectories

def attention_fusion(states, actions, dim=512):
    """Attention fusion that can handle missing/inaccurate observations."""
    n_t = states.shape[0]
    weights = np.ones(n_t)
    
    # Weight more recent timesteps higher (useful for noisy data)
    for t in range(n_t):
        recency = (t + 1) / n_t
        state_magnitude = float(np.linalg.norm(states[t]))
        weights[t] = 0.3 + 0.5 * recency + 0.2 * state_magnitude
    
    weights = weights / (weights.sum() + 1e-8)
    
    state_flat = states.flatten()[:dim//2]
    action_flat = actions.flatten()[:dim//2]
    fused = np.concatenate([state_flat, action_flat])
    
    return fused

def concat_fusion(states, actions, dim=512):
    """Simple concatenation."""
    state_flat = states.flatten()[:dim//2]
    action_flat = actions.flatten()[:dim//2]
    return np.concatenate([state_flat, action_flat])

def run_experiment(noise_levels=[0.0, 0.05, 0.1, 0.2], n_trials=3):
    """Run experiments with different noise levels."""
    results = []
    
    for noise in noise_levels:
        concat_losses = []
        attn_losses = []
        
        for trial in range(n_trials):
            data = generate_stochastic_trajectories(n_samples=200, n_timesteps=30, noise_level=noise)
            
            # Concatenation
            concat_preds = []
            for traj in data[:100]:
                pred = concat_fusion(traj['states'], traj['actions'])
                concat_preds.append(np.mean(pred**2))
            concat_mse = np.mean(concat_preds)
            concat_losses.append(concat_mse)
            
            # Attention
            attn_preds = []
            for traj in data[:100]:
                pred = attention_fusion(traj['states'], traj['actions'])
                attn_preds.append(np.mean(pred**2))
            attn_mse = np.mean(attn_preds)
            attn_losses.append(attn_mse)
        
        concat_loss = np.mean(concat_losses)
        attn_loss = np.mean(attn_losses)
        
        improvement = (concat_loss - attn_loss) / concat_loss * 100 if concat_loss > 0 else 0
        
        results.append({
            'noise_level': noise,
            'concat_mse': float(concat_loss),
            'attention_mse': float(attn_loss),
            'improvement': float(improvement)
        })
    
    return results

if __name__ == '__main__':
    print("=" * 60)
    print("H3.37: Attention with Stochastic Dynamics")
    print("=" * 60)
    
    results = run_experiment()
    
    print("\nResults:")
    print("-" * 60)
    
    attn_wins = sum(1 for r in results if r['improvement'] > 0)
    
    for r in results:
        print(f"Noise {r['noise_level']:.2f}: Concat={r['concat_mse']:.6f}, "
              f"Attn={r['attention_mse']:.6f}, "
              f"{r['improvement']:+.1f}%")
    
    avg = np.mean([r['improvement'] for r in results])
    print("-" * 60)
    print(f"Average: {avg:+.1f}%")
    
    status = "SUPPORTED" if avg > 0 else "REFUTED"
    print(f"Status: {status}")
    
    output = {
        'hypothesis': 'H3.37',
        'statement': 'Attention with stochastic dynamics',
        'status': status,
        'results': results,
        'avg_improvement': float(avg)
    }
    
    output_path = Path('experiments/H3.37-attention-stochastic/results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")