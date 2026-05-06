"""
H1.115: Ultra-Complex Multi-Step Tasks (200-300 Steps)
Tests attention on extreme complexity with compositional reasoning.
"""

import numpy as np
import json
import time

def generate_ultra_complex_task(length, num_subtasks=4):
    """Generate ultra-complex task with multiple subtasks."""
    np.random.seed(int(time.time() * 1000) % 1000000 + length)
    
    state_dim = 16
    action_dim = 16
    
    states = []
    actions = []
    targets = []
    
    subtask_length = length // num_subtasks
    
    for subtask in range(num_subtasks):
        base_pos = np.random.randn(state_dim) * 0.5
        goal_offset = np.random.randn(state_dim) * 0.3
        
        for step in range(subtask_length):
            t = step / subtask_length
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

def crosta_model(states, actions):
    """State Transition Attention (CroSTA)."""
    seq_len = states.shape[0]
    x = np.concatenate([states, actions], axis=-1)
    
    transitions = np.abs(np.diff(x, axis=0, prepend=x[:1]))
    transition_weights = transitions.sum(axis=-1, keepdims=True)
    transition_weights = transition_weights / (transition_weights.max() + 1e-8)
    
    Q = x @ np.random.randn(x.shape[-1], 32)
    K = x @ np.random.randn(x.shape[-1], 32)
    V = x @ np.random.randn(x.shape[-1], 32)
    
    scores = Q @ K.T / np.sqrt(32)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
    attn = attn * (0.5 + 0.5 * transition_weights)
    
    x_attn = attn @ V
    return np.tanh(x_attn @ np.random.randn(x_attn.shape[-1], 16) * 0.1)

def hierarchical_model(states, actions, chunk_size=40):
    """Hierarchical attention model."""
    seq_len = states.shape[0]
    x = np.concatenate([states, actions], axis=-1)
    hidden_dim = 32
    
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    chunk_reprs = []
    
    for c in range(num_chunks):
        start = c * chunk_size
        end = min((c+1) * chunk_size, seq_len)
        chunk = x[start:end]
        
        chunk_repr = chunk.mean(axis=0)
        chunk_reprs.append(chunk_repr)
    
    chunk_reprs = np.array(chunk_reprs)
    
    Q = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], hidden_dim)
    K = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], hidden_dim)
    V = chunk_reprs @ np.random.randn(chunk_reprs.shape[-1], hidden_dim)
    
    scores = Q @ K.T / np.sqrt(hidden_dim)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
    
    chunk_attn = attn @ V
    
    result = []
    for c in range(num_chunks):
        start = c * chunk_size
        end = min((c+1) * chunk_size, seq_len)
        
        local_q = x[start:end] @ np.random.randn(x.shape[-1], hidden_dim)
        local_k = chunk_attn[c:c+1].repeat(end-start, axis=0)
        
        local_scores = (local_q * local_k).sum(axis=-1, keepdims=True)
        local_attn = np.exp(local_scores - np.max(local_scores, axis=-2, keepdims=True))
        local_attn = local_attn / (local_attn.sum(axis=-2, keepdims=True) + 1e-8)
        
        local_v = x[start:end] @ np.random.randn(x.shape[-1], 16) * 0.1
        result.append(local_attn * local_v)
    
    x_hier = np.concatenate(result, axis=0)
    return np.tanh(x_hier)

def compute_loss(pred, target):
    return np.mean((pred - target) ** 2)

def run_experiment():
    results = {}
    improvements = {}
    
    lengths = [180, 200, 240, 280, 320]
    
    for length in lengths:
        states, actions, targets = generate_ultra_complex_task(length, num_subtasks=4)
        
        baseline_pred = baseline_model(states, actions)
        baseline_loss = compute_loss(baseline_pred, targets)
        
        attn_pred = attention_model(states, actions)
        attn_loss = compute_loss(attn_pred, targets)
        
        crosta_pred = crosta_model(states, actions)
        crosta_loss = compute_loss(crosta_pred, targets)
        
        hier_pred = hierarchical_model(states, actions)
        hier_loss = compute_loss(hier_pred, targets)
        
        results[f'baseline_{length}'] = float(baseline_loss)
        results[f'attention_{length}'] = float(attn_loss)
        results[f'crosta_{length}'] = float(crosta_loss)
        results[f'hierarchical_{length}'] = float(hier_loss)
        
        attn_imp = (baseline_loss - attn_loss) / baseline_loss * 100
        crosta_imp = (baseline_loss - crosta_loss) / baseline_loss * 100
        hier_imp = (baseline_loss - hier_loss) / baseline_loss * 100
        
        improvements[str(length)] = {
            'attention': attn_imp,
            'crosta': crosta_imp,
            'hierarchical': hier_imp
        }
    
    avg_attn = np.mean([improvements[l]['attention'] for l in improvements])
    avg_crosta = np.mean([improvements[l]['crosta'] for l in improvements])
    avg_hier = np.mean([improvements[l]['hierarchical'] for l in improvements])
    
    summary = {
        'avg_attention': avg_attn,
        'avg_crosta': avg_crosta,
        'avg_hierarchical': avg_hier,
        'status': 'SUPPORTED' if avg_attn > 30 else 'REFUTED'
    }
    
    output = {
        'experiment': 'H1.115',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'results': results,
        'improvements': improvements,
        'summary': summary
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"H1.115 Results:")
    print(f"  Attention: {avg_attn:.1f}%")
    print(f"  CroSTA: {avg_crosta:.1f}%")
    print(f"  Hierarchical: {avg_hier:.1f}%")
    print(f"  Status: {summary['status']}")
    
    return output

if __name__ == '__main__':
    run_experiment()