"""
H1.424: Hybrid Cognitive Graph Architecture - Simplified version for testing

Design a hybrid architecture that uses:
- Per-Object CG for short-horizon tasks (≤20 steps)
- 2-Node CG for long-horizon tasks (>20 steps)

Test adaptive node selection based on sequence length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import sys

def generate_synthetic_sequence_data(n_demos=1500, seq_len=15, obs_dim=400, lang_dim=32, action_dim=7):
    """Generate synthetic sequence data for testing."""
    np.random.seed(42)
    
    # Generate random data
    observations = np.random.randn(n_demos, seq_len, obs_dim).astype(np.float32)
    language = np.random.randn(n_demos, lang_dim).astype(np.float32)
    
    # Generate actions as a function of observations and language
    # Simple linear relationship with noise
    actions = np.zeros((n_demos, seq_len, action_dim), dtype=np.float32)
    
    for i in range(n_demos):
        # Create a simple dynamic system
        W_obs = np.random.randn(obs_dim, action_dim).astype(np.float32) * 0.1
        W_lang = np.random.randn(lang_dim, action_dim).astype(np.float32) * 0.05
        
        for t in range(seq_len):
            # Action depends on current observation and language
            action = (observations[i, t] @ W_obs + language[i] @ W_lang)
            
            # Add some temporal dependency
            if t > 0:
                action = action * 0.7 + actions[i, t-1] * 0.3
            
            # Add noise
            action += np.random.randn(action_dim).astype(np.float32) * 0.01
            
            actions[i, t] = action
    
    return {
        'observations': observations,
        'language': language,
        'actions': actions
    }

class HybridCognitiveGraph(nn.Module):
    """
    Hybrid Cognitive Graph that adapts its node structure based on sequence length.
    
    For seq_len ≤ 20: Uses per-object nodes (n_objects nodes)
    For seq_len > 20: Uses 2-node abstraction (robot + world nodes)
    """
    
    def __init__(self, obs_dim=400, lang_dim=32, action_dim=7, hidden_dim=64, n_objects=5, 
                 crossover_threshold=20):
        super().__init__()
        self.obs_dim = obs_dim
        self.lang_dim = lang_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_objects = n_objects
        self.crossover_threshold = crossover_threshold
        
        # Shared encoders
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        
        # Per-object CG components (for short horizon)
        self.per_object_node_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        self.per_object_edge_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        self.per_object_node_processor = nn.GRUCell(hidden_dim, hidden_dim)
        self.per_object_edge_processor = nn.GRUCell(hidden_dim, hidden_dim)
        self.per_object_action_decoder = nn.Linear(hidden_dim * n_objects, action_dim)
        
        # 2-Node CG components (for long horizon)
        self.two_node_robot_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        self.two_node_world_encoder = nn.Linear(hidden_dim * 2, hidden_dim)
        self.two_node_robot_processor = nn.GRUCell(hidden_dim, hidden_dim)
        self.two_node_world_processor = nn.GRUCell(hidden_dim, hidden_dim)
        self.two_node_action_decoder = nn.Linear(hidden_dim * 2, action_dim)
        
        # Adaptive selection layer (learns which architecture to use)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64),  # +1 for seq_len feature
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs_seq, lang_emb, seq_len):
        """
        Forward pass with adaptive architecture selection.
        
        Args:
            obs_seq: [batch_size, seq_len, obs_dim]
            lang_emb: [batch_size, lang_dim]
            seq_len: scalar or [batch_size] tensor of sequence lengths
        """
        batch_size = obs_seq.shape[0]
        
        # Encode observations and language
        obs_encoded = self.obs_encoder(obs_seq)  # [batch, seq_len, hidden]
        lang_encoded = self.lang_encoder(lang_emb).unsqueeze(1)  # [batch, 1, hidden]
        lang_encoded_expanded = lang_encoded.expand(-1, obs_encoded.shape[1], -1)
        
        # Combine observation and language features
        combined = torch.cat([obs_encoded, lang_encoded_expanded], dim=-1)  # [batch, seq_len, hidden*2]
        
        # Prepare sequence length feature for selector
        if isinstance(seq_len, torch.Tensor):
            seq_len_feature = seq_len.float().unsqueeze(-1) / 50.0  # Normalize
        else:
            seq_len_feature = torch.tensor([seq_len / 50.0], device=obs_seq.device).expand(batch_size, 1)
        
        # Get architecture selection weights
        # Use first timestep features + seq_len to decide
        selector_input = torch.cat([combined[:, 0, :], seq_len_feature], dim=-1)
        selection_weights = self.selector(selector_input)  # [batch, 2]
        
        # Process with both architectures
        per_object_output = self._per_object_forward(combined)
        two_node_output = self._two_node_forward(combined)
        
        # Weighted combination based on selector
        per_object_weight = selection_weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        two_node_weight = selection_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        
        # Combine outputs
        combined_output = per_object_weight * per_object_output + two_node_weight * two_node_output
        
        # Also return selection weights for analysis
        return combined_output, selection_weights
    
    def _per_object_forward(self, combined_features):
        """Forward pass through per-object CG architecture."""
        batch_size, seq_len, _ = combined_features.shape
        
        # Initialize node states (one per object)
        node_states = torch.zeros(batch_size, self.n_objects, self.hidden_dim, 
                                 device=combined_features.device)
        
        # Initialize edge states (fully connected)
        edge_states = torch.zeros(batch_size, self.n_objects, self.n_objects, self.hidden_dim,
                                 device=combined_features.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current combined features
            current = combined_features[:, t, :]  # [batch, hidden*2]
            
            # Update nodes
            node_inputs = current.unsqueeze(1).expand(-1, self.n_objects, -1)
            node_updates = self.per_object_node_encoder(node_inputs)
            node_states = self.per_object_node_processor(node_updates.view(-1, self.hidden_dim), 
                                                        node_states.view(-1, self.hidden_dim))
            node_states = node_states.view(batch_size, self.n_objects, self.hidden_dim)
            
            # Update edges (simplified - no message passing for efficiency)
            edge_inputs = torch.cat([
                node_states.unsqueeze(2).expand(-1, -1, self.n_objects, -1),
                node_states.unsqueeze(1).expand(-1, self.n_objects, -1, -1)
            ], dim=-1)
            edge_updates = self.per_object_edge_encoder(edge_inputs)
            edge_states = self.per_object_edge_processor(
                edge_updates.view(-1, self.hidden_dim),
                edge_states.view(-1, self.hidden_dim)
            )
            edge_states = edge_states.view(batch_size, self.n_objects, self.n_objects, self.hidden_dim)
            
            # Decode action from all node states
            node_features_flat = node_states.view(batch_size, -1)
            action = self.per_object_action_decoder(node_features_flat)
            outputs.append(action.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)
    
    def _two_node_forward(self, combined_features):
        """Forward pass through 2-node CG architecture."""
        batch_size, seq_len, _ = combined_features.shape
        
        # Initialize node states (robot and world)
        robot_state = torch.zeros(batch_size, self.hidden_dim, device=combined_features.device)
        world_state = torch.zeros(batch_size, self.hidden_dim, device=combined_features.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current combined features
            current = combined_features[:, t, :]  # [batch, hidden*2]
            
            # Update robot node
            robot_input = self.two_node_robot_encoder(current)
            robot_state = self.two_node_robot_processor(robot_input, robot_state)
            
            # Update world node
            world_input = self.two_node_world_encoder(current)
            world_state = self.two_node_world_processor(world_input, world_state)
            
            # Decode action from both nodes
            combined_state = torch.cat([robot_state, world_state], dim=-1)
            action = self.two_node_action_decoder(combined_state)
            outputs.append(action.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

class BaselineMLP(nn.Module):
    """Baseline MLP for comparison."""
    
    def __init__(self, obs_dim=400, lang_dim=32, action_dim=7, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lang_encoder = nn.Linear(lang_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs_seq, lang_emb, seq_len):
        batch_size, seq_len, _ = obs_seq.shape
        
        # Encode
        obs_encoded = self.obs_encoder(obs_seq)  # [batch, seq_len, hidden]
        lang_encoded = self.lang_encoder(lang_emb).unsqueeze(1)  # [batch, 1, hidden]
        lang_encoded_expanded = lang_encoded.expand(-1, seq_len, -1)
        
        # Combine
        combined = torch.cat([obs_encoded, lang_encoded_expanded], dim=-1)
        
        # Process
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """Train a model and return validation metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for obs_seq, lang_emb, actions, seq_len in train_loader:
            obs_seq, lang_emb, actions = obs_seq.to(device), lang_emb.to(device), actions.to(device)
            
            optimizer.zero_grad()
            if isinstance(model, HybridCognitiveGraph):
                pred_actions, selection_weights = model(obs_seq, lang_emb, seq_len)
            else:
                pred_actions = model(obs_seq, lang_emb, seq_len)
            
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for obs_seq, lang_emb, actions, seq_len in val_loader:
                obs_seq, lang_emb, actions = obs_seq.to(device), lang_emb.to(device), actions.to(device)
                
                if isinstance(model, HybridCognitiveGraph):
                    pred_actions, _ = model(obs_seq, lang_emb, seq_len)
                else:
                    pred_actions = model(obs_seq, lang_emb, seq_len)
                
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set and return MSE and MAE."""
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    selection_stats = []
    
    with torch.no_grad():
        for obs_seq, lang_emb, actions, seq_len in test_loader:
            obs_seq, lang_emb, actions = obs_seq.to(device), lang_emb.to(device), actions.to(device)
            
            if isinstance(model, HybridCognitiveGraph):
                pred_actions, selection_weights = model(obs_seq, lang_emb, seq_len)
                # Collect selection statistics
                for i in range(len(seq_len)):
                    selection_stats.append({
                        'seq_len': seq_len[i].item(),
                        'per_object_weight': selection_weights[i, 0].item(),
                        'two_node_weight': selection_weights[i, 1].item()
                    })
            else:
                pred_actions = model(obs_seq, lang_emb, seq_len)
            
            mse = criterion_mse(pred_actions, actions)
            mae = criterion_mae(pred_actions, actions)
            
            total_mse += mse.item() * len(obs_seq)
            total_mae += mae.item() * len(obs_seq)
            total_samples += len(obs_seq)
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae, selection_stats

def analyze_selection_stats(selection_stats, seq_len):
    """Analyze the architecture selection patterns."""
    if not selection_stats:
        return {"error": "No selection stats available"}
    
    # Calculate average weights
    per_object_avg = np.mean([s['per_object_weight'] for s in selection_stats])
    two_node_avg = np.mean([s['two_node_weight'] for s in selection_stats])
    
    # Determine preferred architecture
    preferred = "per_object" if per_object_avg > two_node_avg else "two_node"
    
    # Check if selection aligns with crossover threshold (20 steps)
    expected_preferred = "per_object" if seq_len <= 20 else "two_node"
    alignment = "aligned" if preferred == expected_preferred else "misaligned"
    
    # Calculate confidence (difference between weights)
    confidence = abs(per_object_avg - two_node_avg)
    
    return {
        'per_object_weight_avg': per_object_avg,
        'two_node_weight_avg': two_node_avg,
        'preferred_architecture': preferred,
        'expected_preferred': expected_preferred,
        'alignment': alignment,
        'selection_confidence': confidence,
        'seq_len': seq_len
    }

def run_experiment(config):
    """Run the hybrid architecture experiment."""
    print(f"Running experiment with config: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_sequence_data(
        n_demos=config['n_demos'],
        seq_len=config['seq_len'],
        obs_dim=config['obs_dim'],
        lang_dim=config['lang_dim'],
        action_dim=config['action_dim']
    )
    
    # Split data
    train_size = int(0.7 * len(data['observations']))
    val_size = int(0.15 * len(data['observations']))
    
    train_obs = torch.tensor(data['observations'][:train_size], dtype=torch.float32)
    train_lang = torch.tensor(data['language'][:train_size], dtype=torch.float32)
    train_actions = torch.tensor(data['actions'][:train_size], dtype=torch.float32)
    train_seq_len = torch.tensor([config['seq_len']] * train_size, dtype=torch.float32)
    
    val_obs = torch.tensor(data['observations'][train_size:train_size+val_size], dtype=torch.float32)
    val_lang = torch.tensor(data['language'][train_size:train_size+val_size], dtype=torch.float32)
    val_actions = torch.tensor(data['actions'][train_size:train_size+val_size], dtype=torch.float32)
    val_seq_len = torch.tensor([config['seq_len']] * val_size, dtype=torch.float32)
    
    test_obs = torch.tensor(data['observations'][train_size+val_size:], dtype=torch.float32)
    test_lang = torch.tensor(data['language'][train_size+val_size:], dtype=torch.float32)
    test_actions = torch.tensor(data['actions'][train_size+val_size:], dtype=torch.float32)
    test_seq_len = torch.tensor([config['seq_len']] * (len(data['observations']) - train_size - val_size), dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(train_obs, train_lang, train_actions, train_seq_len)
    val_dataset = TensorDataset(val_obs, val_lang, val_actions, val_seq_len)
    test_dataset = TensorDataset(test_obs, test_lang, test_actions, test_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Data generated: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Initialize models
    print("Initializing models...")
    hybrid_model = HybridCognitiveGraph(
        obs_dim=config['obs_dim'],
        lang_dim=config['lang_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        n_objects=config['n_objects'],
        crossover_threshold=20
    )
    
    baseline_model = BaselineMLP(
        obs_dim=config['obs_dim'],
        lang_dim=config['lang_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    # Train and evaluate hybrid model
    print("\nTraining hybrid model...")
    hybrid_train_loss, hybrid_val_loss = train_model(
        hybrid_model, train_loader, val_loader,
        epochs=config['epochs'], lr=config['lr'], device=device
    )
    
    print("\nEvaluating hybrid model...")
    hybrid_mse, hybrid_mae, selection_stats = evaluate_model(hybrid_model, test_loader, device=device)
    
    # Train and evaluate baseline
    print("\nTraining baseline model...")
    baseline_train_loss, baseline_val_loss = train_model(
        baseline_model, train_loader, val_loader,
        epochs=config['epochs'], lr=config['lr'], device=device
    )
    
    print("\nEvaluating baseline model...")
    baseline_mse, baseline_mae, _ = evaluate_model(baseline_model, test_loader, device=device)
    
    # Calculate improvement
    improvement = ((baseline_mse - hybrid_mse) / baseline_mse) * 100
    
    # Analyze selection statistics
    selection_analysis = analyze_selection_stats(selection_stats, config['seq_len'])
    
    # Prepare results
    results = {
        'config': config,
        'hybrid_model': {
            'test_mse': hybrid_mse,
            'test_mae': hybrid_mae,
            'train_loss': hybrid_train_loss[-1] if hybrid_train_loss else None,
            'val_loss': hybrid_val_loss[-1] if hybrid_val_loss else None
        },
        'baseline_model': {
            'test_mse': baseline_mse,
            'test_mae': baseline_mae,
            'train_loss': baseline_train_loss[-1] if baseline_train_loss else None,
            'val_loss': baseline_val_loss[-1] if baseline_val_loss else None
        },
        'improvement': {
            'mse_improvement_percent': improvement,
            'absolute_mse_difference': baseline_mse - hybrid_mse
        },
        'selection_analysis': selection_analysis,
        'selection_stats_sample': selection_stats[:10] if selection_stats else []
    }
    
    print(f"\nResults:")
    print(f"  Hybrid Model MSE: {hybrid_mse:.6f}")
    print(f"  Baseline Model MSE: {baseline_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Selection Analysis: {selection_analysis}")
    
    return results

def main():
    """Main experiment function."""
    # Experiment configuration
    config = {
        'experiment_id': 'H1.424',
        'description': 'Hybrid Cognitive Graph Architecture - Adaptive node selection based on sequence length',
        'n_demos': 1500,
        'seq_len': 15,  # Test at crossover point
        'n_objects': 5,
        'obs_dim': 400,
        'lang_dim': 32,
        'action_dim': 7,
        'hidden_dim': 64,
        'epochs': 20,
        'lr': 0.001,
        'batch_size': 32,
        'n_runs': 1,
        'timestamp': datetime.now().isoformat()
    }
    
    print("=" * 80)
    print(f"Experiment: {config['experiment_id']}")
    print(f"Description: {config['description']}")
    print("=" * 80)
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    os.makedirs('../results', exist_ok=True)
    results_file = f"../results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()