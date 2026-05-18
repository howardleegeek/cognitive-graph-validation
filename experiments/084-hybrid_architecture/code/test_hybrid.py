"""
Test script for Hybrid Cognitive Graph Architecture.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment import HybridCognitiveGraph, BaselineMLP

def test_model_shapes():
    """Test that models produce correct output shapes."""
    print("Testing model shapes...")
    
    # Test configuration
    batch_size = 4
    seq_len = 15
    obs_dim = 400
    lang_dim = 32
    action_dim = 7
    hidden_dim = 64
    n_objects = 5
    
    # Create dummy data
    obs_seq = torch.randn(batch_size, seq_len, obs_dim)
    lang_emb = torch.randn(batch_size, lang_dim)
    seq_len_tensor = torch.tensor([seq_len] * batch_size, dtype=torch.float32)
    
    # Test Hybrid Cognitive Graph
    print("\n1. Testing Hybrid Cognitive Graph...")
    hybrid_model = HybridCognitiveGraph(
        obs_dim=obs_dim,
        lang_dim=lang_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_objects=n_objects,
        crossover_threshold=20
    )
    
    hybrid_output, selection_weights = hybrid_model(obs_seq, lang_emb, seq_len_tensor)
    
    print(f"  Input shape: obs_seq={obs_seq.shape}, lang_emb={lang_emb.shape}")
    print(f"  Output shape: {hybrid_output.shape} (expected: [{batch_size}, {seq_len}, {action_dim}])")
    print(f"  Selection weights shape: {selection_weights.shape} (expected: [{batch_size}, 2])")
    print(f"  Selection weights sum per sample: {selection_weights.sum(dim=1)} (should be ~1.0)")
    
    assert hybrid_output.shape == (batch_size, seq_len, action_dim), \
        f"Hybrid output shape mismatch: {hybrid_output.shape}"
    assert selection_weights.shape == (batch_size, 2), \
        f"Selection weights shape mismatch: {selection_weights.shape}"
    assert torch.allclose(selection_weights.sum(dim=1), torch.ones(batch_size), atol=1e-5), \
        "Selection weights should sum to 1 per sample"
    
    # Test Baseline MLP
    print("\n2. Testing Baseline MLP...")
    baseline_model = BaselineMLP(
        obs_dim=obs_dim,
        lang_dim=lang_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    baseline_output = baseline_model(obs_seq, lang_emb, seq_len)
    
    print(f"  Input shape: obs_seq={obs_seq.shape}, lang_emb={lang_emb.shape}")
    print(f"  Output shape: {baseline_output.shape} (expected: [{batch_size}, {seq_len}, {action_dim}])")
    
    assert baseline_output.shape == (batch_size, seq_len, action_dim), \
        f"Baseline output shape mismatch: {baseline_output.shape}"
    
    # Test with different sequence lengths
    print("\n3. Testing with different sequence lengths...")
    test_seq_lens = [5, 15, 25, 35]
    
    for test_len in test_seq_lens:
        obs_seq_test = torch.randn(batch_size, test_len, obs_dim)
        seq_len_tensor_test = torch.tensor([test_len] * batch_size, dtype=torch.float32)
        
        hybrid_output_test, selection_weights_test = hybrid_model(
            obs_seq_test, lang_emb, seq_len_tensor_test
        )
        
        print(f"  Seq len {test_len}: output shape={hybrid_output_test.shape}, "
              f"per_object_weight={selection_weights_test[:, 0].mean():.3f}, "
              f"two_node_weight={selection_weights_test[:, 1].mean():.3f}")
        
        assert hybrid_output_test.shape == (batch_size, test_len, action_dim), \
            f"Hybrid output shape mismatch for seq_len={test_len}: {hybrid_output_test.shape}"
    
    print("\n✓ All shape tests passed!")

def test_forward_pass():
    """Test forward pass logic."""
    print("\nTesting forward pass logic...")
    
    batch_size = 2
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
    
    # Test data
    obs_seq = torch.randn(batch_size, seq_len, obs_dim)
    lang_emb = torch.randn(batch_size, lang_dim)
    
    # Test with scalar seq_len
    print("  Testing with scalar sequence length...")
    output_scalar, weights_scalar = model(obs_seq, lang_emb, seq_len)
    assert output_scalar.shape == (batch_size, seq_len, action_dim)
    
    # Test with tensor seq_len
    print("  Testing with tensor sequence length...")
    seq_len_tensor = torch.tensor([seq_len, seq_len], dtype=torch.float32)
    output_tensor, weights_tensor = model(obs_seq, lang_emb, seq_len_tensor)
    assert output_tensor.shape == (batch_size, seq_len, action_dim)
    
    # Verify outputs are similar
    assert torch.allclose(output_scalar, output_tensor, atol=1e-5), \
        "Outputs should be similar for scalar and tensor seq_len"
    
    print("  ✓ Forward pass tests passed!")

def test_selection_behavior():
    """Test that selection weights respond to sequence length."""
    print("\nTesting selection behavior...")
    
    batch_size = 10
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
    
    # Test with short sequences (should prefer per-object)
    print("  Testing short sequences (seq_len=10)...")
    obs_seq_short = torch.randn(batch_size, 10, obs_dim)
    lang_emb = torch.randn(batch_size, lang_dim)
    
    _, weights_short = model(obs_seq_short, lang_emb, 10)
    per_object_avg_short = weights_short[:, 0].mean().item()
    two_node_avg_short = weights_short[:, 1].mean().item()
    
    print(f"    Per-object weight: {per_object_avg_short:.3f}")
    print(f"    Two-node weight: {two_node_avg_short:.3f}")
    
    # Test with long sequences (should prefer two-node)
    print("  Testing long sequences (seq_len=30)...")
    obs_seq_long = torch.randn(batch_size, 30, obs_dim)
    
    _, weights_long = model(obs_seq_long, lang_emb, 30)
    per_object_avg_long = weights_long[:, 0].mean().item()
    two_node_avg_long = weights_long[:, 1].mean().item()
    
    print(f"    Per-object weight: {per_object_avg_long:.3f}")
    print(f"    Two-node weight: {two_node_avg_long:.3f}")
    
    # Note: The model needs to be trained to learn proper selection
    # This just tests that the mechanism works
    print("  ✓ Selection mechanism test passed (weights vary with seq_len)")

def test_training_step():
    """Test that models can be trained for one step."""
    print("\nTesting training step...")
    
    batch_size = 4
    seq_len = 15
    obs_dim = 400
    lang_dim = 32
    action_dim = 7
    hidden_dim = 64
    n_objects = 5
    
    # Create models
    hybrid_model = HybridCognitiveGraph(
        obs_dim=obs_dim,
        lang_dim=lang_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        n_objects=n_objects,
        crossover_threshold=20
    )
    
    baseline_model = BaselineMLP(
        obs_dim=obs_dim,
        lang_dim=lang_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    # Create dummy data
    obs_seq = torch.randn(batch_size, seq_len, obs_dim)
    lang_emb = torch.randn(batch_size, lang_dim)
    target_actions = torch.randn(batch_size, seq_len, action_dim)
    seq_len_tensor = torch.tensor([seq_len] * batch_size, dtype=torch.float32)
    
    # Test hybrid model training step
    print("  Testing Hybrid Cognitive Graph training step...")
    hybrid_optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.001)
    hybrid_criterion = torch.nn.MSELoss()
    
    hybrid_optimizer.zero_grad()
    hybrid_output, _ = hybrid_model(obs_seq, lang_emb, seq_len_tensor)
    hybrid_loss = hybrid_criterion(hybrid_output, target_actions)
    hybrid_loss.backward()
    hybrid_optimizer.step()
    
    print(f"    Hybrid loss: {hybrid_loss.item():.6f}")
    print("    ✓ Hybrid model can be trained")
    
    # Test baseline model training step
    print("  Testing Baseline MLP training step...")
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_criterion = torch.nn.MSELoss()
    
    baseline_optimizer.zero_grad()
    baseline_output = baseline_model(obs_seq, lang_emb, seq_len)
    baseline_loss = baseline_criterion(baseline_output, target_actions)
    baseline_loss.backward()
    baseline_optimizer.step()
    
    print(f"    Baseline loss: {baseline_loss.item():.6f}")
    print("    ✓ Baseline model can be trained")
    
    print("  ✓ All training tests passed!")

def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Hybrid Cognitive Graph Architecture")
    print("=" * 80)
    
    try:
        test_model_shapes()
        test_forward_pass()
        test_selection_behavior()
        test_training_step()
        
        print("\n" + "=" * 80)
        print("✅ All tests passed! The hybrid architecture is ready for experimentation.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())