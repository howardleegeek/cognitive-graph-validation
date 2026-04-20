"""
H3.2: Test graph-enhanced attention on complex temporal tasks
Hypothesis: Graph-enhanced attention outperforms concatenation on temporal tasks
"""
import numpy as np
from sklearn.neural_network import MLPRegressor


def generate_temporal_data(n_samples, n_timesteps=12, seed=42):
    np.random.seed(seed)
    state_dim = 24
    action_dim = 8
    X, y = [], []
    for _ in range(n_samples):
        states = []
        actions = []
        state = np.random.randn(state_dim) * 0.5
        for t in range(n_timesteps):
            action = np.random.randn(action_dim) * 0.2
            action_expanded = np.tile(action, 3)[:state_dim]
            next_state = state + action_expanded * 0.15 + np.random.randn(state_dim) * 0.02
            states.append(state)
            actions.append(action)
            state = next_state
        X.append(np.concatenate([s for s in states] + [a for a in actions]))
        y.append(state)
    return np.array(X), np.array(y)


def run_concat(X, y, test_X, test_y, dim=2048):
    """Standard concatenation approach"""
    model = MLPRegressor(hidden_layer_sizes=(dim, dim // 2), activation='relu',
                         solver='adam', max_iter=500, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    model.fit(X, y)
    pred = model.predict(test_X)
    return np.mean((pred - test_y) ** 2)


def run_graph_attention(X, y, test_X, test_y, dim=2048):
    """Graph-enhanced with attention-like weighting"""
    n_samples = X.shape[0]
    total_features = X.shape[1]
    
    # Compute dimensions from data
    # X = states[0], states[1], ..., states[n-1], actions[0], ..., actions[n-1]
    # Need to find n (timesteps)
    state_dim = 24
    action_dim = 8
    # Features = n * state_dim + n * action_dim = n * (state_dim + action_dim)
    seq_len = total_features // (state_dim + action_dim)
    
    # Enhance with graph structure + attention-like features
    X_enhanced = []
    for i in range(n_samples):
        sample = X[i]
        states = sample[:seq_len * state_dim].reshape(seq_len, state_dim)
        actions = sample[seq_len * state_dim:].reshape(seq_len, action_dim)
        
        # Graph features: state differences (temporal edges)
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        
        # Attention-like: weighted recent states
        weights = np.exp(np.arange(seq_len) * 0.1)  # More weight on recent
        weights = weights / weights.sum()
        weighted_state = np.sum(states * weights[:, np.newaxis], axis=0)
        
        # Combined: original + graph + attention
        enhanced = np.concatenate([
            sample,
            diffs_padded.flatten(),
            weighted_state,
            np.sum(actions * weights[:, np.newaxis], axis=0)  # weighted actions
        ])
        X_enhanced.append(enhanced)
    X_enhanced = np.array(X_enhanced)
    
    # Increase input dim for model
    model = MLPRegressor(hidden_layer_sizes=(dim, dim // 2), activation='relu',
                         solver='adam', max_iter=600, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    model.fit(X_enhanced, y)
    
    # Enhance test data
    test_enhanced = []
    test_total = test_X.shape[1]
    test_seq_len = test_total // (state_dim + action_dim)
    for i in range(test_X.shape[0]):
        sample = test_X[i]
        states = sample[:test_seq_len * state_dim].reshape(test_seq_len, state_dim)
        actions = sample[test_seq_len * state_dim:].reshape(test_seq_len, action_dim)
        
        diffs = np.diff(states, axis=0)
        diffs_padded = np.vstack([np.zeros((1, state_dim)), diffs])
        
        weights = np.exp(np.arange(seq_len) * 0.1)
        weights = weights / weights.sum()
        weighted_state = np.sum(states * weights[:, np.newaxis], axis=0)
        
        enhanced = np.concatenate([
            sample,
            diffs_padded.flatten(),
            weighted_state,
            np.sum(actions * weights[:, np.newaxis], axis=0)
        ])
        test_enhanced.append(enhanced)
    test_enhanced = np.array(test_enhanced)
    
    pred = model.predict(test_enhanced)
    return np.mean((pred - test_y) ** 2)


def run_experiment():
    print("\n=== H3.2: Graph Attention vs Concatenation on Temporal Tasks ===")
    
    train_X, train_y = generate_temporal_data(400, n_timesteps=12, seed=42)
    test_X, test_y = generate_temporal_data(200, n_timesteps=12, seed=789)
    
    print("\n12-step temporal tasks:")
    concat_loss = run_concat(train_X, train_y, test_X, test_y)
    print(f"Concatenation: {concat_loss:.4f}")
    
    graph_attn_loss = run_graph_attention(train_X, train_y, test_X, test_y)
    print(f"Graph + Attention: {graph_attn_loss:.4f}")
    
    improvement = (concat_loss - graph_attn_loss) / concat_loss * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    print("\n16-step temporal tasks:")
    train_X16, train_y16 = generate_temporal_data(400, n_timesteps=16, seed=42)
    test_X16, test_y16 = generate_temporal_data(200, n_timesteps=16, seed=789)
    
    concat_loss16 = run_concat(train_X16, train_y16, test_X16, test_y16)
    print(f"Concatenation: {concat_loss16:.4f}")
    
    graph_attn_loss16 = run_graph_attention(train_X16, train_y16, test_X16, test_y16)
    print(f"Graph + Attention: {graph_attn_loss16:.4f}")
    
    improvement16 = (concat_loss16 - graph_attn_loss16) / concat_loss16 * 100
    print(f"\nImprovement: {improvement16:.1f}%")
    
    results = {
        "12step": {"concat": concat_loss, "graph_attn": graph_attn_loss, "improvement": improvement},
        "16step": {"concat": concat_loss16, "graph_attn": graph_attn_loss16, "improvement": improvement16}
    }
    print("\n=== Summary ===")
    print(f"12-step: Concat={concat_loss:.4f}, Graph+Attn={graph_attn_loss:.4f}, delta={improvement:+.1f}%")
    print(f"16-step: Concat={concat_loss16:.4f}, Graph+Attn={graph_attn_loss16:.4f}, delta={improvement16:+.1f}%")
    return results


if __name__ == '__main__':
    results = run_experiment()