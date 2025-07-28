"""
Phase 1 Testing for HRM Gating Mechanism
Tests basic functionality, gradient flow, and backward compatibility
"""

import torch
import torch.nn.functional as F
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.layers import HRMGatingNetwork


def test_gating_network_basic():
    """Test basic functionality of HRMGatingNetwork."""
    print("Testing HRMGatingNetwork basic functionality...")
    
    hidden_size = 256
    batch_size = 2
    seq_len = 10
    
    # Create gating network
    gate_net = HRMGatingNetwork(hidden_size)
    
    # Test inputs
    z_L = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    z_H = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    # Forward pass
    g_L, g_H, g_X = gate_net(z_L, z_H, x)
    
    # Check shapes
    assert g_L.shape == z_L.shape, f"Gate L shape mismatch: {g_L.shape} vs {z_L.shape}"
    assert g_H.shape == z_H.shape, f"Gate H shape mismatch: {g_H.shape} vs {z_H.shape}"
    assert g_X.shape == x.shape, f"Gate X shape mismatch: {g_X.shape} vs {x.shape}"
    
    # Check gate values are in [0, 1]
    assert torch.all(g_L >= 0) and torch.all(g_L <= 1), "Gate L values out of range"
    assert torch.all(g_H >= 0) and torch.all(g_H <= 1), "Gate H values out of range"
    assert torch.all(g_X >= 0) and torch.all(g_X <= 1), "Gate X values out of range"
    
    # Test gradient flow
    loss = (g_L.sum() + g_H.sum() + g_X.sum())
    loss.backward()
    
    # Check gradients exist and are not NaN
    for name, param in gate_net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"
        assert not torch.any(torch.isinf(param.grad)), f"Inf gradient in {name}"
    
    # Check input gradients
    assert z_L.grad is not None and not torch.any(torch.isnan(z_L.grad)), "Bad gradient for z_L"
    assert z_H.grad is not None and not torch.any(torch.isnan(z_H.grad)), "Bad gradient for z_H"
    assert x.grad is not None and not torch.any(torch.isnan(x.grad)), "Bad gradient for x"
    
    print("✓ HRMGatingNetwork basic tests passed")
    
    # Print gate statistics
    print(f"  Gate means - L: {g_L.mean():.3f}, H: {g_H.mean():.3f}, X: {g_X.mean():.3f}")
    print(f"  Gate stds  - L: {g_L.std():.3f}, H: {g_H.std():.3f}, X: {g_X.std():.3f}")


def test_gated_model_forward():
    """Test full model with gating enabled."""
    print("\nTesting gated HRM model forward pass...")
    
    # Configuration with gating enabled
    config_dict = {
        'batch_size': 2,
        'seq_len': 32,
        'vocab_size': 100,
        'num_puzzle_identifiers': 10,
        'H_cycles': 2,
        'L_cycles': 3,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 3,
        'halt_exploration_prob': 0.1,
        'use_gating': True,  # Enable gating
        'gate_hidden_ratio': 0.25,
        'gate_init_bias': -2.2
    }
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(config_dict)
    model.eval()  # Set to eval mode for testing
    
    # Create dummy batch
    batch = {
        'inputs': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'labels': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'puzzle_identifiers': torch.randint(0, config_dict['num_puzzle_identifiers'], (config_dict['batch_size'],))
    }
    
    # Test forward pass
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    
    # Verify outputs
    assert 'logits' in outputs, "Missing logits in outputs"
    assert outputs['logits'].shape == (config_dict['batch_size'], config_dict['seq_len'], config_dict['vocab_size']), \
        f"Wrong logits shape: {outputs['logits'].shape}"
    
    # Check for NaN/Inf
    assert not torch.any(torch.isnan(outputs['logits'])), "NaN in logits"
    assert not torch.any(torch.isinf(outputs['logits'])), "Inf in logits"
    
    print("✓ Gated model forward pass test passed")


def test_backward_compatibility():
    """Test that models work with gating disabled (backward compatibility)."""
    print("\nTesting backward compatibility (gating disabled)...")
    
    # Configuration with gating disabled
    config_dict = {
        'batch_size': 2,
        'seq_len': 32,
        'vocab_size': 100,
        'num_puzzle_identifiers': 10,
        'H_cycles': 2,
        'L_cycles': 3,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 3,
        'halt_exploration_prob': 0.1,
        'use_gating': False  # Disable gating
    }
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(config_dict)
    model.eval()
    
    # Create dummy batch
    batch = {
        'inputs': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'labels': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'puzzle_identifiers': torch.randint(0, config_dict['num_puzzle_identifiers'], (config_dict['batch_size'],))
    }
    
    # Test forward pass
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    
    # Verify outputs
    assert 'logits' in outputs, "Missing logits in outputs"
    assert outputs['logits'].shape == (config_dict['batch_size'], config_dict['seq_len'], config_dict['vocab_size']), \
        f"Wrong logits shape: {outputs['logits'].shape}"
    
    print("✓ Backward compatibility test passed")


def test_gradient_flow_full_model():
    """Test gradient flow through the entire gated model."""
    print("\nTesting gradient flow through gated model...")
    
    # Configuration
    config_dict = {
        'batch_size': 2,
        'seq_len': 16,
        'vocab_size': 50,
        'num_puzzle_identifiers': 5,
        'H_cycles': 1,
        'L_cycles': 2,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 64,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'use_gating': True
    }
    
    # Create model
    model = HierarchicalReasoningModel_ACTV1(config_dict)
    model.train()
    
    # Create dummy batch
    batch = {
        'inputs': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'labels': torch.randint(0, config_dict['vocab_size'], (config_dict['batch_size'], config_dict['seq_len'])),
        'puzzle_identifiers': torch.randint(0, config_dict['num_puzzle_identifiers'], (config_dict['batch_size'],))
    }
    
    # Forward pass
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    
    # Compute loss
    logits = outputs['logits']
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check gradients in gating networks
    gating_params_checked = 0
    for name, param in model.named_parameters():
        if 'gating_network' in name and param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"
            assert not torch.any(torch.isinf(param.grad)), f"Inf gradient in {name}"
            gating_params_checked += 1
    
    assert gating_params_checked > 0, "No gating parameters found!"
    print(f"✓ Gradient flow test passed ({gating_params_checked} gating parameters checked)")


def compare_gated_vs_ungated():
    """Compare outputs between gated and ungated models to ensure different behavior."""
    print("\nComparing gated vs ungated model outputs...")
    
    # Base configuration
    base_config = {
        'batch_size': 2,
        'seq_len': 16,
        'vocab_size': 50,
        'num_puzzle_identifiers': 5,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 1,
        'L_layers': 1,
        'hidden_size': 64,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
    }
    
    # Create gated model
    config_gated = {**base_config, 'use_gating': True}
    model_gated = HierarchicalReasoningModel_ACTV1(config_gated)
    
    # Create ungated model
    config_ungated = {**base_config, 'use_gating': False}
    model_ungated = HierarchicalReasoningModel_ACTV1(config_ungated)
    
    # Set to eval mode
    model_gated.eval()
    model_ungated.eval()
    
    # Create dummy batch
    torch.manual_seed(42)  # For reproducibility
    batch = {
        'inputs': torch.randint(0, base_config['vocab_size'], (base_config['batch_size'], base_config['seq_len'])),
        'labels': torch.randint(0, base_config['vocab_size'], (base_config['batch_size'], base_config['seq_len'])),
        'puzzle_identifiers': torch.randint(0, base_config['num_puzzle_identifiers'], (base_config['batch_size'],))
    }
    
    # Forward pass for both models
    carry_gated = model_gated.initial_carry(batch)
    carry_ungated = model_ungated.initial_carry(batch)
    
    _, outputs_gated = model_gated(carry_gated, batch)
    _, outputs_ungated = model_ungated(carry_ungated, batch)
    
    # Compare outputs - they should be different due to gating
    logits_diff = torch.abs(outputs_gated['logits'] - outputs_ungated['logits'])
    mean_diff = logits_diff.mean().item()
    
    print(f"  Mean absolute difference in logits: {mean_diff:.6f}")
    
    # The outputs should be different (but not too different since gates start near 0.1)
    assert mean_diff > 0, "Gated and ungated models produce identical outputs!"
    assert mean_diff < 10, f"Outputs differ too much: {mean_diff}"
    
    print("✓ Gated vs ungated comparison test passed")


def run_all_tests():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Running Phase 1 Gating Mechanism Tests")
    print("=" * 60)
    
    try:
        test_gating_network_basic()
        test_gated_model_forward()
        test_backward_compatibility()
        test_gradient_flow_full_model()
        compare_gated_vs_ungated()
        
        print("\n" + "=" * 60)
        print("✅ All Phase 1 tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1) 