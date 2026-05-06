"""
H1.116: Adaptive Attention Switching
Tests adaptive architecture that switches between attention and concatenation based on sequence length.
"""

import numpy as np
import json
import time

def generate_task(length):
    """Generate task with varying complexity."""
    np.random.seed(int(time.time() * 1000) % 1000000 + length)
    
    state_dim = 16
    action_dim = 16
    
    states = []
    actions = []
    targets = []
    
    base_pos = np.random.randn(state_dim) * 0.5
    goal_offset = np.random.randn(state_dim) * 0.3
    
    for step in range(length):
        t = step / max(length, 1)
        progress = np.sin(t * np.pi)
        
        state = base_pos + goal_offset * progress + np.random.randn(state_dim) * 0.1
        action = goal_offset * progress + np.random.randn(action_dim) * 0.05
        
        target = base_pos + goal_offset + np.random.randn(state_dim) * 0.05
        
        states.append(state)
        actions.append(action)
        targets.append(target)
    
    return np.array(states), np.array(actions), np.array(targets)

def baseline_model(states, actions):
    """Concatenation baseline."""
    x = np.concatenate([states, actions], axis=-1)
    return np.tanh(x @ np.random.randn(x.shape[-1], 16) * 0.1)

def attention_model(states, actions, num_heads=4):
    """Multi-head attention model."""
    seq_len = states.shape[0]
    x = np.concatenate([states, actions], axis=-1)
    
    Q = x @ np.random.randn(x.shape[-1], 32)
    K = x @ np.random.randn(x.shape[-1], 32)
    V = x @ np.random.randn(x.shape[-1], 32)
    
    head_dim = 32 // num_heads
    outputs = []
    for h in range(num_heads):
        Q_h = Q[:, h*head_dim:(h+1)*head_dim]
        K_h = K[:, h*head_dim:(h+1)*head_dim]
        V_h = V[:, h*head_dim:(h+1)*head_dim]
        
        scores = Q_h @ K_h.T / np.sqrt(head_dim)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        outputs.append(attn @ V_h)
    
    x_attn = np.concatenate(outputs, axis=-1)
    return np.tanh(x_attn @ np.random.randn(x_attn.shape[-1], 16) * 0.1)

def adaptive_model(states, actions, threshold=150):
    """Adaptive attention switching model."""
    seq_len = states.shape[0]
    x = np.concatenate([states, actions], axis=-1)
    
    if seq_len < threshold:
        Q = x @ np.random.randn(x.shape[-1], 32)
        K = x @ np.random.randn(x.shape[-1], 32)
        V = x @ np.random.randn(x.shape[-1], 32)
        
        scores = Q @ K.T / np.sqrt(32)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        x_adapt = attn @ V
        x_adapt = x_adapt @ np.random.randn(x_adapt.shape[-1], 16) * 0.1
    else:
        chunk_size = 50
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunk_reprs = []
        
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c+1) * chunk_size, seq_len)
            chunk = x[start:end]
            chunk_repr = chunk.mean(axis=0)
            chunk_reprs.append(chunk_repr)
        
        chunk_reprs = np.array(chunk_reprs)
        
        Q = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], 24)
        K = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], 24)
        V = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], 24)
        
        scores = Q @ K.T / np.sqrt(24)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        chunk_attn = attn @ V
        
        result = []
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c+1) * chunk_size, seq_len)
            
            local_q = x[start:end] @ np.random.randn(x.shape[-1], 24)
            local_k = chunk_attn[c:c+1].repeat(end-start, axis=0)
            
            local_scores = (local_q * local_k).sum(axis=-1, keepdims=True)
            local_attn = np.exp(local_scores - np.max(local_scores, axis=-2, keepdims=True))
            local_attn = local_attn / (local_attn.sum(axis=-2, keepdims=True) + 1e-8)
            
            local_v = x[start:end] @ np.random.randn(x.shape[-1], 16) * 0.1
            result.append(local_attn * local_v)
        
        x_adapt = np.concatenate(result, axis=0)
    
    return np.tanh(x_adapt)

def compute_loss(pred, target):
    return np.mean((pred - target) ** 2)

def run_experiment():
    results = {}
    improvements = {}
    
    lengths = [50, 80, 100, 120, 150, 180, 200, 250]
    
    for length in lengths:
        states, actions, targets = generate_task(length)
        
        baseline_pred = baseline_model(states, actions)
        baseline_loss = compute_loss(baseline_pred, targets)
        
        attn_pred = attention_model(states, actions)
        attn_loss = compute_loss(attn_pred, targets)
        
        adaptive_pred = adaptive_model(states, actions, threshold=150)
        adaptive_loss = compute_loss(adaptive_pred, targets)
        
        results[f'baseline_{length}'] = float(baseline_loss)
        results[f'attention_{length}'] = float(attn_loss)
        results[f'adaptive_{length}'] = float(adaptive_loss)
        
        attn_imp = (baseline_loss - attn_loss) / baseline_loss * 100
        adaptive_imp = (baseline_loss - adaptive_loss) / baseline_loss * 100
        
        improvements[str(length)] = {
            'attention': attn_imp,
            'adaptive': adaptive_imp
        }
    
    short_lengths = [l for l in lengths if l < 150]
    long_lengths = [l for l in lengths if l >= 150]
    
    avg_attn_short = np.mean([improvements[str(l)]['attention'] for l in short_lengths])
    avg_adaptive_short = np.mean([improvements[str(l)]['adaptive'] for l in short_lengths])
    
    avg_attn_long = np.mean([improvements[str(l)]['attention'] for l in long_lengths])
    avg_adaptive_long = np.mean([improvements[str(l)]['adaptive'] for l in long_lengths])
    
    avg_attn_all = np.mean([improvements[str(l)]['attention'] for l in lengths])
    avg_adaptive_all = np.mean([improvements[str(l)]['adaptive'] for l in lengths])
    
    status = 'SUPPORTED' if (avg_adaptive_short > avg_attn_short * 0.9 and avg_adaptive_long > avg_attn_long) else 'REFUTED'
    
    summary = {
        'avg_attn_short': avg_attn_short,
        'avg_adaptive_short': avg_adaptive_short,
        'avg_attn_long': avg_attn_long,
        'avg_adaptive_long': avg_adaptive_long,
        'avg_attention': avg_attn_all,
        'avg_adaptive': avg_adaptive_all,
        'status': status
    }
    
    output = {
        'experiment': 'H1.116',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'results': results,
        'improvements': improvements,
        'summary': summary
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"H1.116 Results:")
    print(f"  Short (<150): Attention {avg_attn_short:.1f}%, Adaptive {avg_adaptive_short:.1f}%")
    print(f"  Long (>=150): Attention {avg_attn_long:.1f}%, Adaptive {avg_adaptive_long:.1f}%")
    print(f"  Overall: Attention {avg_attn_all:.1f}%, Adaptive {avg_adaptive_all:.1f}%")
    print(f"  Status: {status}")
    
    return output

if __name__ == '__main__':
    run_experiment()