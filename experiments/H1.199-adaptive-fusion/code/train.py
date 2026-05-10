"""
H1.199: Adaptive Fusion Architecture - Switches between concat/attention/SSM based on task complexity
Based on findings:
- H1.198: Attention wins at 50-100 steps with high autocorrelation (0.85)
- H1.193: SSM wins +97.6% on 50-step with next-step prediction
- H1.195: Baseline wins 20-80 steps (final-step prediction)
- H3.76: SSM+Attention hybrid outperforms both on real robot
- H1.73: Hybrid task-adaptive achieves +79.6%

Hypothesis: Learned complexity detector can route between architectures more effectively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("H1.199: Adaptive Fusion Architecture")
print("=" * 60)

# Generate synthetic data with varying complexity
def generate_task_data(n_samples=500, max_steps=30, autocorr=0.85):
    """Generate synthetic robot-like task data with fixed-length sequences"""
    X_physical = []
    X_semantic = []
    y = []
    
    for _ in range(n_samples):
        # Fixed sequence length
        seq_len = max_steps + np.random.randint(-5, 5)
        seq_len = max(10, min(seq_len, max_steps * 2))  # Clamp to valid range
        
        # Generate sequence with temporal structure
        physical = np.zeros((seq_len, 16), dtype=np.float32)
        semantic = np.zeros((seq_len, 16), dtype=np.float32)
        
        state = np.random.randn(16)
        last_semantic = np.zeros(16)
        
        for t in range(seq_len):
            # Add temporal autocorrelation (robot-like behavior)
            state = autocorr * state + (1 - autocorr) * np.random.randn(16)
            
            # Physical: position, velocity, force
            physical[t] = state + np.random.randn(16) * 0.1
            
            # Semantic: task phase, object IDs, instructions
            semantic[t] = 0.8 * last_semantic + 0.2 * np.random.randn(16)
            last_semantic = semantic[t]
        
        X_physical.append(physical)
        X_semantic.append(semantic)
        y.append(state + np.random.randn(16) * 0.01)
    
    return X_physical, X_semantic, np.array(y)

# Simple concatenation baseline
class ConcatBaseline(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
    
    def forward(self, x_physical, x_semantic):
        x = torch.cat([x_physical, x_semantic], dim=-1)
        out = self.encoder(x)
        return out[:, -1]  # Return last timestep

# Standard attention mechanism
class AttentionFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.q_proj = nn.Linear(16, 64)
        self.k_proj = nn.Linear(16, 64)
        self.v_proj = nn.Linear(16, 64)
        self.out_proj = nn.Linear(64, 16)
        self.encoder = nn.Sequential(
            nn.Linear(32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
        
    def forward(self, x_physical, x_semantic, mask=None):
        # Temporal attention over sequence
        seq_len = x_physical.shape[1]
        
        # Cross-modal attention (project to common dim)
        q = self.q_proj(x_semantic)      # (B, T, 64)
        k = self.k_proj(x_physical)      # (B, T, 64)
        v = self.v_proj(x_physical)      # (B, T, 64)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / 8.0  # (B, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        
        # Context: weighted sum over physical
        context = torch.matmul(attn, v)   # (B, T, 64)
        
        # Project context to 16
        context = self.out_proj(context)  # (B, T, 16)
        
        # Concat context with semantic (32 dim)
        combined = torch.cat([context, x_semantic], dim=-1)  # (B, T, 32)
        
        return self.encoder(combined)[:, -1]  # (B, 16)

# SSM module (simplified Mamba-style)
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projection
        self.x_proj = nn.Linear(d_model, d_state * 2, bias=True)
        
        # SSM parameters (learned)
        self.A = nn.Parameter(torch.randn(d_state, d_model))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Output projection
        self.out_proj = nn.Linear(d_state, d_model)
        
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # Input-dependent parameters
        x_bdt = self.x_proj(x)  # (B, T, d_state*2)
        B_t = torch.tanh(x_bdt[:, :, :self.d_state])  # (B, T, d_state)
        
        # SSM scan
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, d_model)
            b_t = B_t[:, t, :]  # (B, d_state)
            
            # Discretization
            dt = F.softplus((b_t @ self.A) / self.d_state)  # (B, d_model)
            dA = torch.exp(-dt)  # Decay
            
            # Update
            h_new = dA * h + (1 - dA) * b_t
            y_t = self.out_proj(h_new) + self.D * x_t
            outputs.append(y_t)
            h = h_new
        
        return torch.stack(outputs, dim=1)  # (B, T, d_model)

# SSM fusion model
class SSMFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.physical_proj = nn.Linear(16, 64)
        self.semantic_proj = nn.Linear(16, 64)
        
        # Simple RNN-style SSM
        self.rnn = nn.GRU(128, 128, batch_first=True)
        
        self.out = nn.Sequential(
            nn.Linear(128, dim),
            nn.ReLU(),
            nn.Linear(dim, 16)
        )
        
    def forward(self, x_physical, x_semantic):
        p = self.physical_proj(x_physical)
        s = self.semantic_proj(x_semantic)
        combined = torch.cat([p, s], dim=-1)
        
        # RNN processing
        out, _ = self.rnn(combined)
        
        return self.out(out[:, -1])  # Last step

# Learned complexity detector
class ComplexityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Complexity score
            nn.Sigmoid()
        )
        
        # Per-step encoder
        self.step_encoder = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        
    def forward(self, x_physical, x_semantic):
        x = torch.cat([x_physical, x_semantic], dim=-1)
        
        # Global complexity
        global_complexity = self.encoder(x.mean(dim=1))
        
        # Temporal complexity (variance over time)
        step_features, _ = self.step_encoder(x)
        temporal_complexity = step_features.std(dim=1).mean(-1, keepdim=True)
        temporal_complexity = torch.sigmoid(temporal_complexity / 10)
        
        # Sequence length factor
        seq_len = x.shape[1]
        length_factor = torch.tensor([min(seq_len / 100.0, 1.0)], device=x.device).view(1, 1)
        
        # Combined complexity
        combined = (global_complexity + temporal_complexity + length_factor) / 3.0
        
        return combined

# Adaptive router that selects architecture
class AdaptiveRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.complexity_detector = ComplexityDetector()
        
        # Architecture selection head
        self.selection_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 architectures
        )
        
    def forward(self, x_physical, x_semantic):
        complexity = self.complexity_detector(x_physical, x_semantic)
        logits = self.selection_net(complexity)
        
        # Soft selection (probabilities)
        probs = torch.softmax(logits, dim=-1)
        
        return probs, complexity

# Combined adaptive fusion model
class AdaptiveFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.concat_model = ConcatBaseline()
        self.attention_model = AttentionFusion()
        self.ssm_model = SSMFusion()
        self.router = AdaptiveRouter()
        
    def forward(self, x_physical, x_semantic, training=True):
        batch_size = x_physical.shape[0]
        
        # Get selection probabilities
        probs, complexity = self.router(x_physical, x_semantic)
        
        # Get predictions from each model
        concat_pred = self.concat_model(x_physical, x_semantic)
        attn_pred = self.attention_model(x_physical, x_semantic)
        ssm_pred = self.ssm_model(x_physical, x_semantic)
        
        # Stack predictions
        all_preds = torch.stack([concat_pred, attn_pred, ssm_pred], dim=1)  # (B, 3, 16)
        
        if training:
            # Soft weighted combination during training
            weights = probs.unsqueeze(-1)  # (B, 3, 1)
            prediction = (all_preds * weights).sum(dim=1)  # (B, 16)
        else:
            # Hard selection during inference
            idx = probs.argmax(dim=-1)  # (B,)
            prediction = all_preds[torch.arange(batch_size, device=all_preds.device), idx]
        
        return prediction, probs, complexity

# Hard selection variant (baseline)
class HardSelectionModel(nn.Module):
    def __init__(self, thresholds={'concat': 0.3, 'attention': 0.6}):
        super().__init__()
        self.concat_model = ConcatBaseline()
        self.attention_model = AttentionFusion()
        self.ssm_model = SSMFusion()
        self.thresholds = thresholds
        
    def forward(self, x_physical, x_semantic):
        # Simple heuristic-based selection
        x = torch.cat([x_physical, x_semantic], dim=-1)
        complexity = x.std().item()
        
        if complexity < self.thresholds['concat']:
            return self.concat_model(x_physical, x_semantic)
        elif complexity < self.thresholds['attention']:
            return self.attention_model(x_physical, x_semantic)
        else:
            return self.ssm_model(x_physical, x_semantic)

def train_and_evaluate(model, train_data, val_data, epochs=100, lr=0.001, name="Model"):
    """Train model and evaluate"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_train_phys, X_train_sem, y_train = train_data
    X_val_phys, X_val_sem, y_val = val_data
    
    # Convert to tensors - pad to same length
    max_len = max(max(len(p) for p in X_train_phys), max(len(p) for p in X_val_phys))
    
    def pad_sequences(phys_list, sem_list, max_len):
        phys_padded = np.zeros((len(phys_list), max_len, 16), dtype=np.float32)
        sem_padded = np.zeros((len(sem_list), max_len, 16), dtype=np.float32)
        for i, (p, s) in enumerate(zip(phys_list, sem_list)):
            l = len(p)
            phys_padded[i, :l] = p
            sem_padded[i, :l] = s
        return torch.tensor(phys_padded).to(device), torch.tensor(sem_padded).to(device)
    
    train_phys, train_sem = pad_sequences(X_train_phys, X_train_sem, max_len)
    val_phys, val_sem = pad_sequences(X_val_phys, X_val_sem, max_len)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if hasattr(model, 'router'):
            prediction, probs, complexity = model(train_phys, train_sem, training=True)
        else:
            prediction = model(train_phys, train_sem)
        
        loss = criterion(prediction, train_y)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'router'):
                val_pred, _, _ = model(val_phys, val_sem, training=False)
            else:
                val_pred = model(val_phys, val_sem)
            val_loss = criterion(val_pred, val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    # Restore best
    model.load_state_dict(best_state)
    
    return best_val_loss

def main():
    results = {}
    
    print("\n[1] Generating training data...")
    # Test on varying complexity tasks
    test_configs = [
        {'name': 'Simple (10-20 steps)', 'n_steps': 15, 'autocorr': 0.7},
        {'name': 'Medium (20-40 steps)', 'n_steps': 30, 'autocorr': 0.85},
        {'name': 'Complex (40-60 steps)', 'n_steps': 50, 'autocorr': 0.9},
        {'name': 'Very Complex (60-100 steps)', 'n_steps': 80, 'autocorr': 0.95},
    ]
    
    print("\n[2] Training and evaluating models...")
    
    all_concat_mse = []
    all_attn_mse = []
    all_ssm_mse = []
    all_adaptive_mse = []
    all_hard_mse = []
    
    for config in test_configs:
        print(f"\n  --- {config['name']} ---")
        
        # Generate data
        train_data = generate_task_data(n_samples=400, max_steps=config['n_steps'], autocorr=config['autocorr'])
        val_data = generate_task_data(n_samples=100, max_steps=config['n_steps'], autocorr=config['autocorr'])
        
        # Train individual models
        print("    Training Concat Baseline...")
        concat_model = ConcatBaseline()
        concat_mse = train_and_evaluate(concat_model, train_data, val_data, name="Concat")
        
        print("    Training Attention Model...")
        attn_model = AttentionFusion()
        attn_mse = train_and_evaluate(attn_model, train_data, val_data, name="Attention")
        
        print("    Training SSM Model...")
        ssm_model = SSMFusion()
        ssm_mse = train_and_evaluate(ssm_model, train_data, val_data, name="SSM")
        
        print("    Training Adaptive Fusion...")
        adaptive_model = AdaptiveFusionModel()
        adaptive_mse = train_and_evaluate(adaptive_model, train_data, val_data, name="Adaptive")
        
        print("    Training Hard Selection...")
        hard_model = HardSelectionModel()
        hard_mse = train_and_evaluate(hard_model, train_data, val_data, name="Hard")
        
        all_concat_mse.append(concat_mse)
        all_attn_mse.append(attn_mse)
        all_ssm_mse.append(ssm_mse)
        all_adaptive_mse.append(adaptive_mse)
        all_hard_mse.append(hard_mse)
        
        print(f"    Results: Concat={concat_mse:.6f}, Attn={attn_mse:.6f}, SSM={ssm_mse:.6f}")
        print(f"             Adaptive={adaptive_mse:.6f}, Hard={hard_mse:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n| Task Complexity | Concat | Attention | SSM | Adaptive | Hard |")
    print("|-----------------|--------|-----------|-----|----------|------|")
    
    for i, config in enumerate(test_configs):
        print(f"| {config['name']:17} | {all_concat_mse[i]:.6f} | {all_attn_mse[i]:.9f} | {all_ssm_mse[i]:.6f} | {all_adaptive_mse[i]:.6f} | {all_hard_mse[i]:.6f} |")
    
    # Calculate improvements
    adaptive_vs_concat = [(c - a) / c * 100 for c, a in zip(all_concat_mse, all_adaptive_mse)]
    adaptive_vs_hard = [(h - a) / h * 100 for h, a in zip(all_hard_mse, all_adaptive_mse)]
    
    print("\n| Adaptive vs Concat | Adaptive vs Hard |")
    print("|--------------------|------------------|")
    for i, config in enumerate(test_configs):
        print(f"| {adaptive_vs_concat[i]:+.2f}% | {adaptive_vs_hard[i]:+.2f}% |")
    
    avg_improvement_concat = np.mean(adaptive_vs_concat)
    avg_improvement_hard = np.mean(adaptive_vs_hard)
    
    print(f"\nAverage: Adaptive vs Concat = {avg_improvement_concat:+.2f}%")
    print(f"Average: Adaptive vs Hard = {avg_improvement_hard:+.2f}%")
    
    # Determine status
    if avg_improvement_concat > 0 and avg_improvement_hard > 0:
        status = "SUPPORTED"
        print(f"\n✅ H1.199: {status} - Adaptive fusion outperforms both fixed and hard selection!")
    elif avg_improvement_concat > 0:
        status = "SUPPORTED (marginal)"
        print(f"\n⚠️ H1.199: {status} - Adaptive fusion helps vs concat only")
    else:
        status = "REFUTED"
        print(f"\n❌ H1.199: {status} - No clear benefit from adaptive fusion")
    
    # Store results
    results['status'] = status
    results['concat'] = all_concat_mse
    results['attention'] = all_attn_mse
    results['ssm'] = all_ssm_mse
    results['adaptive'] = all_adaptive_mse
    results['hard'] = all_hard_mse
    results['improvement_concat'] = adaptive_vs_concat
    results['improvement_hard'] = adaptive_vs_hard
    
    return results

if __name__ == "__main__":
    results = main()
    
    print("\n" + "=" * 60)
    print("H1.199 COMPLETE")
    print("=" * 60)