"""
Simple test script for Hybrid Cognitive Graph Architecture.
"""

import torch
import torch.nn as nn
import numpy as np

class HybridCognitiveGraph(nn.Module):
    """
    Hybrid Cognitive Graph that adapts its node structure based on sequence length.
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
            
            # Update edges (simplified)
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

def test_basic_functionality():
    """Test basic functionality of the hybrid architecture."""
    print("Testing Hybrid Cognitive Graph Architecture...")
    print("=" * 60)
    
    # Test configuration
    batch_size = 4
    seq_len = 15
    obs_dim = 400
    lang_dim = 32
    action_dim = 7
    hidden_dim = 64
    n_objects = 5
    
    # Create model
    model = HybridCognitiveGraph(
        obs_dim=obs_dim,
        lang_dim=lang_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_objects=n_objects,
        crossover_threshold=20
    )
    
    # Create dummy data
    obs_seq = torch.randn(batch_size, seq_len, obs_dim)
    lang_emb = torch.randn(batch_size, lang_dim)
    seq_len_tensor = torch.tensor([seq_len] * batch_size, dtype=torch.float32)
    
    # Test forward pass
    print("\n1. Testing forward pass...")
    output, selection_weights = model(obs_seq, lang_emb, seq_len_tensor)
    
    print(f"  Input shapes:")
    print(f"    obs_seq: {obs_seq.shape}")
    print(f"    lang_emb: {lang_emb.shape}")
    print(f"    seq_len: {seq_len_tensor.shape}")
    
    print(f"\n  Output shapes:")
    print(f"    output: {output.shape} (expected: [{batch_size}, {seq_len}, {action_dim}])")
    print(f"    selection_weights: {selection_weights.shape} (expected: [{batch_size}, 2])")
    
    # Verify shapes
    assert output.shape == (batch_size, seq_len, action_dim), \
        f"Output shape mismatch: {output.shape}"
    assert selection_weights.shape == (batch_size, 2), \
        f"Selection weights shape mismatch: {selection_weights.shape}"
    
    # Verify selection weights sum to 1
    weights_sum = selection_weights.sum(dim=1)
    print(f"\n  Selection weights sum per sample: {weights_sum}")
    assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-5), \
        "Selection weights should sum to 1 per sample"
    
    print("  ✓ Forward pass test passed!")
    
    # Test with different sequence lengths
    print("\n2. Testing with different sequence lengths...")
    test_lengths = [5, 10, 20, 30]
    
    for test_len in test_lengths:
        obs_seq_test = torch.randn(batch_size, test_len, obs_dim)
        seq_len_tensor_test = torch.tensor([test_len] * batch_size, dtype=torch.float32)
        
        output_test, weights_test = model(obs_seq_test, lang_emb, seq_len_tensor_test)
        
        print(f"  Seq len {test_len:2d}: output shape={output_test.shape}, "
              f"per_object_weight={weights_test[:, 0].mean():.3f}, "
              f"two_node_weight={weights_test[:, 1].mean():.3f}")
        
        assert output_test.shape == (batch_size, test_len, action_dim), \
            f"Output shape mismatch for seq_len={test_len}: {output_test.shape}"
    
    print("  ✓ Variable sequence length test passed!")
    
    # Test training step
    print("\n3. Testing training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create target actions
    target_actions = torch.randn(batch_size, seq_len, action_dim)
    
    # Training step
    optimizer.zero_grad()
    pred_actions, _ = model(obs_seq, lang_emb, seq_len_tensor)
    loss = criterion(pred_actions, target_actions)
    loss.backward()
    optimizer.step()
    
    print(f"  Loss after one training step: {loss.item():.6f}")
    
    # Check gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients after backward pass"
    print("  ✓ Training step test passed!")
    
    # Test model parameters
    print("\n4. Testing model parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    print("  ✓ Model parameters test passed!")
    
    print("\n" + "=" * 60)
    print("✅ All basic functionality tests passed!")
    print("=" * 60)

def test_selection_mechanism():
    """Test that the selection mechanism works correctly."""
    print("\nTesting selection mechanism...")
    print("=" * 60)
    
    # Create a simple test model
    model = HybridCognitiveGraph(
        obs_dim=10,  # Smaller for testing
        lang_dim=8,
        action_dim=3,
        hidden_dim=16,
        n_objects=3,
        crossover_threshold=20
    )
    
    # Test that selector produces valid outputs
    batch_size = 8
    
    # Test with short sequence
    print("\n1. Testing with short sequence (seq_len=10)...")
    obs_seq_short = torch.randn(batch_size, 10, 10)
    lang_emb = torch.randn(batch_size, 8)
    
    output_short, weights_short = model(obs_seq_short, lang_emb, 10)
    
    print(f"  Average per-object weight: {weights_short[:, 0].mean():.3f}")
    print(f"  Average two-node weight: {weights_short[:, 1].mean():.3f}")
    
    # Test with long sequence
    print("\n2. Testing with long sequence (seq_len=30)...")
    obs_seq_long = torch.randn(batch_size, 30, 10)
    
    output_long, weights_long = model(obs_seq_long, lang_emb, 30)
    
    print(f"  Average per-object weight: {weights_long[:, 0].mean():.3f}")
    print(f"  Average two-node weight: {weights_long[:, 1].mean():.3f}")
    
    # Note: The model needs to be trained to learn proper selection
    # This just tests that the mechanism produces valid outputs
    print("\n  ✓ Selection mechanism produces valid outputs")
    
    # Test that weights are in [0, 1] range
    assert torch.all(weights_short >= 0) and torch.all(weights_short <= 1), \
        "Selection weights should be in [0, 1] range"
    assert torch.all(weights_long >= 0) and torch.all(weights_long <= 1), \
        "Selection weights should be in [0, 1] range"
    
    print("  ✓ Selection weights are in valid range")
    
    print("\n" + "=" * 60)
    print("✅ Selection mechanism tests passed!")
    print("=" * 60)

def main():
    """Run all tests."""
    print("Hybrid Cognitive Graph Architecture - Basic Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_selection_mechanism()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The hybrid architecture is ready for experimentation.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())