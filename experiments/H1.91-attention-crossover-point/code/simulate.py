"""
H1.91: Attention Crossover Point Discovery

Comparing attention-weighted vs direct concatenation on temporal prediction.
Based on prior results (findings.md):
- H3.4: attention wins at 24, 30 steps (marginal -0.4%)
- H3.6: +100% on 40+ steps  
- H3.7: +99.6% on 300-1000 timesteps

This simulates weighted sum of history (attention) vs flat concatenation.
"""
import numpy as np
import json
import sys

np.random.seed(42)

def generate_problematic_data(n_samples, n_timesteps, state_dim=16, decay_rate=0.1):
    """
    Generate data where recent history is most predictive.
    This is where attention should shine.
    """
    X = np.random.randn(n_samples, n_timesteps, state_dim).astype(np.float32) * 0.5
    
    y = np.zeros((n_samples, n_timesteps, state_dim))
    for i in range(n_samples):
        for t in range(n_timesteps):
            # y[t] = weighted average of history with exponential decay
            weights = np.exp(-decay_rate * np.arange(t+1))[::-1]  # recent gets higher weight
            weights = weights / weights.sum()
            y[i,t,:] = (weights[:, None] * X[i,:t+1,:]).sum(axis=0) + 0.001 * np.random.randn(state_dim)
    
    return X, y

def predict_last_only(X, y):
    """Baseline: predict y[T] from x[T] only"""
    n_samples = X.shape[0]
    inputs = X[:, -1, :]
    target = y[:, -1, :]
    
    # Linear: y = Wx + b
    W = np.linalg.lstsq(inputs, target, rcond=None)[0]
    pred = inputs @ W
    mse = np.mean((pred - target) ** 2)
    return mse, W

def predict_weighted_history(X, y, decay_rate=0.1):
    """Attention-style: weighted sum of history with learnable decay"""
    n_samples, n_timesteps, state_dim = X.shape
    
    target = y[:, -1, :]
    inputs_list = []
    preds_list = []
    
    for i in range(n_samples):
        # Weighted average of history
        weights = np.exp(-decay_rate * np.arange(n_timesteps))[::-1]
        weights = weights / weights.sum()
        context = (weights[:, None] * X[i,:,:]).sum(axis=0)
        inputs_list.append(context)
        
        # Predict
        W = np.linalg.lstsq(np.array([context]), target[i:i+1], rcond=None)[0]
        preds_list.append(context @ W)
    
    inputs = np.array(inputs_list)
    preds = np.array(preds_list).squeeze()
    mse = np.mean((preds - target) ** 2)
    return mse, decay_rate

def predict_full_concat(X, y):
    """Full concatenation: use all timesteps"""
    n_samples, n_timesteps, state_dim = X.shape
    
    inputs = X.reshape(n_samples, -1)
    target = y[:, -1, :].reshape(n_samples, -1)
    
    W = np.linalg.lstsq(inputs, target, rcond=None)[0]
    pred = inputs @ W
    mse = np.mean((pred - target) ** 2)
    return mse, W

def predict_attention_learned(X, y, n_heads=4):
    """Simulated attention: learn weights that minimize prediction error"""
    n_samples, n_timesteps, state_dim = X.shape
    target = y[:, -1, :]
    
    # Try multiple decay rates and pick best
    best_mse = float('inf')
    best_decay = 0.1
    
    for decay in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        weights = np.exp(-decay * np.arange(n_timesteps))[::-1]
        weights = weights / weights.sum()
        
        contexts = []
        for i in range(n_samples):
            ctx = (weights[:, None] * X[i,:,:]).sum(axis=0)
            contexts.append(ctx)
        
        contexts = np.array(contexts)
        
        # Linear prediction
        try:
            W = np.linalg.lstsq(contexts, target, rcond=None)[0]
            pred = contexts @ W
            mse = np.mean((pred - target) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_decay = decay
        except:
            pass
    
    return best_mse, best_decay

def experiment():
    """Run crossover experiment"""
    results = {
        "n_steps": [],
        "baseline_mse": [],
        "weighted_mse": [],
        "concat_mse": [],
        "learned_attn_mse": [],
        "weighted_vs_baseline": [],
        "weighted_vs_concat": [],
        "learned_vs_concat": []
    }
    
    timesteps_range = [5, 8, 10, 12, 15, 20, 24, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("Testing attention crossover point...")
    sys.stdout.flush()
    
    for n_steps in timesteps_range:
        X, y = generate_problematic_data(200, n_steps, decay_rate=0.1)
        
        # 1. Last only baseline
        base_mse, _ = predict_last_only(X, y)
        
        # 2. Weighted history (fixed decay)
        weighted_mse, decay = predict_weighted_history(X, y)
        
        # 3. Full concat
        concat_mse, _ = predict_full_concat(X, y)
        
        # 4. Learned attention
        learned_mse, best_decay = predict_attention_learned(X, y)
        
        # Deltas
        wb = (base_mse - weighted_mse) / base_mse * 100 if base_mse > 0 else 0
        wc = (concat_mse - weighted_mse) / concat_mse * 100 if concat_mse > 0 else 0
        lc = (concat_mse - learned_mse) / concat_mse * 100 if concat_mse > 0 else 0
        
        results["n_steps"].append(n_steps)
        results["baseline_mse"].append(float(base_mse))
        results["weighted_mse"].append(float(weighted_mse))
        results["concat_mse"].append(float(concat_mse))
        results["learned_attn_mse"].append(float(learned_mse))
        results["weighted_vs_baseline"].append(float(wb))
        results["weighted_vs_concat"].append(float(wc))
        results["learned_vs_concat"].append(float(lc))
        
        winner = "ATTN" if wc > 0 else "CONCAT"
        print(f"  {n_steps:3d}: baseline={base_mse:.4f}, weighted={weighted_mse:.5f}, concat={concat_mse:.5f}, learned={learned_mse:.5f} Δ(learned vs concat)={lc:+6.1f}% [{winner}]")
        sys.stdout.flush()
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("H1.91: Attention Crossover Point Discovery")
    print("=" * 60)
    
    results = experiment()
    
    # Find crossover
    crossover_concat = None
    for i, delta in enumerate(results["learned_vs_concat"]):
        if delta > 0:
            crossover_concat = results["n_steps"][i]
            break
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Crossover (attention beats full concat): ~{crossover_concat} timesteps")
    
    attn_wins = [results["n_steps"][i] for i, d in enumerate(results["learned_vs_concat"]) if d > 0]
    concat_wins = [results["n_steps"][i] for i, d in enumerate(results["learned_vs_concat"]) if d <= 0]
    
    print(f"Attention beats concat at: {attn_wins}")
    print(f"Concat beats attention at: {concat_wins}")
    
    # Average advantage
    avg = np.mean(results["learned_vs_concat"])
    print(f"\nAverage attention advantage: {avg:+.1f}%")
    
    results["crossover_point"] = crossover_concat
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")