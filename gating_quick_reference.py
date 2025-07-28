"""
Quick Reference Implementation for HRM Gating Mechanism
This file contains ready-to-use code snippets for implementing gating in HRM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

# Import existing HRM components
from models.layers import CastedLinear
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Block


class HRMGatingNetwork(nn.Module):
    """Core gating network for controlling information flow."""
    
    def __init__(self, hidden_size: int, num_gates: int = 3):
        super().__init__()
        # Lightweight design: use 1/4 of hidden size for gate computation
        gate_hidden = hidden_size // 4
        
        # Two-layer MLP for gate computation
        self.gate_proj = nn.Sequential(
            CastedLinear(hidden_size * 3, gate_hidden, bias=True),
            nn.SiLU(),  # Smooth activation for gates
            CastedLinear(gate_hidden, num_gates * hidden_size, bias=True)
        )
        
        # Initialize gates to be slightly open (0.1 after sigmoid)
        with torch.no_grad():
            self.gate_proj[-1].bias.fill_(-2.2)
    
    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor, 
                x_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gates for L, H, and input channels."""
        # Concatenate all inputs
        gate_input = torch.cat([z_L, z_H, x_input], dim=-1)
        
        # Compute gates through MLP
        gates = self.gate_proj(gate_input)
        
        # Reshape to separate gates
        gates = gates.view(*gates.shape[:-1], 3, -1)
        
        # Apply sigmoid for [0, 1] range
        gates = torch.sigmoid(gates)
        
        # Split into individual gates
        gate_L, gate_H, gate_X = gates.unbind(dim=-2)
        
        return gate_L, gate_H, gate_X


class HierarchicalReasoningModel_ACTV1ReasoningModule_Gated(nn.Module):
    """Gated version of the reasoning module."""
    
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block], 
                 hidden_size: int, use_gating: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.use_gating = use_gating
        
        if use_gating:
            self.gating_network = HRMGatingNetwork(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, 
                secondary_states: torch.Tensor, 
                input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with gated information flow."""
        
        if self.use_gating:
            # Compute adaptive gates
            gate_primary, gate_secondary, gate_input = self.gating_network(
                hidden_states, secondary_states, input_injection
            )
            
            # Apply gates element-wise
            gated_primary = gate_primary * hidden_states
            gated_secondary = gate_secondary * secondary_states
            gated_input = gate_input * input_injection
            
            # Sum gated components
            hidden_states = gated_primary + gated_secondary + gated_input
        else:
            # Original behavior: simple addition
            hidden_states = hidden_states + secondary_states + input_injection
        
        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        
        return hidden_states


def compute_gating_losses(gate_values: Dict[str, torch.Tensor], 
                         prev_gate_values: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Compute auxiliary losses for training gated models."""
    losses = {}
    
    for name, gates in gate_values.items():
        # 1. Saturation loss - prevent gates from sticking to 0 or 1
        min_dist_to_boundary = torch.min(gates, 1 - gates)
        saturation_loss = torch.mean(min_dist_to_boundary ** 2)
        losses[f'{name}_saturation'] = saturation_loss
        
        # 2. Entropy loss - encourage diverse gating patterns
        # Convert to probabilities across gate dimension
        gate_probs = F.softmax(gates.log(), dim=-1)  # Use log for numerical stability
        entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1)
        losses[f'{name}_entropy'] = -torch.mean(entropy)  # Negative because we maximize entropy
        
        # 3. Stability loss - prevent rapid gate changes (if previous values available)
        if prev_gate_values and name in prev_gate_values:
            stability_loss = torch.mean((gates - prev_gate_values[name]) ** 2)
            losses[f'{name}_stability'] = stability_loss
    
    return losses


class GatingMonitor:
    """Monitor and analyze gating patterns during training."""
    
    def __init__(self):
        self.gate_history = []
    
    def record(self, step: int, gate_values: Dict[str, torch.Tensor]):
        """Record gate values for analysis."""
        stats = {}
        for name, gates in gate_values.items():
            stats[name] = {
                'mean': gates.mean().item(),
                'std': gates.std().item(),
                'min': gates.min().item(),
                'max': gates.max().item(),
                'sparsity': (gates < 0.1).float().mean().item()
            }
        
        self.gate_history.append({
            'step': step,
            'stats': stats
        })
    
    def get_summary(self):
        """Get summary statistics of gating behavior."""
        if not self.gate_history:
            return {}
        
        # Aggregate statistics across training
        summary = {}
        gate_names = list(self.gate_history[0]['stats'].keys())
        
        for name in gate_names:
            summary[name] = {
                'mean_activation': np.mean([h['stats'][name]['mean'] for h in self.gate_history]),
                'mean_sparsity': np.mean([h['stats'][name]['sparsity'] for h in self.gate_history]),
                'activation_trend': [h['stats'][name]['mean'] for h in self.gate_history[-100:]],  # Last 100 steps
            }
        
        return summary


# Example integration into existing forward pass
def gated_forward_example(self, carry, batch):
    """Example of how to modify the existing forward pass for gating."""
    # ... existing setup code ...
    
    # Modified forward iterations with gating
    with torch.no_grad():
        z_H, z_L = carry.z_H, carry.z_L
        
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and 
                       (_L_step == self.config.L_cycles - 1)):
                    # L-module with gated inputs
                    z_L = self.L_level(
                        hidden_states=z_L,
                        secondary_states=z_H,
                        input_injection=input_embeddings,
                        **seq_info
                    )
            
            if not (_H_step == self.config.H_cycles - 1):
                # H-module receives L-state, no direct input injection
                z_H = self.H_level(
                    hidden_states=z_H,
                    secondary_states=z_L,
                    input_injection=torch.zeros_like(z_H),
                    **seq_info
                )
    
    # 1-step gradient computation (same pattern as above)
    z_L = self.L_level(
        hidden_states=z_L,
        secondary_states=z_H,
        input_injection=input_embeddings,
        **seq_info
    )
    z_H = self.H_level(
        hidden_states=z_H,
        secondary_states=z_L,
        input_injection=torch.zeros_like(z_H),
        **seq_info
    )
    
    # ... rest of forward pass ...


# Configuration update example
GATING_CONFIG_DEFAULTS = {
    'use_gating': True,
    'gating_type': 'standard',  # Options: 'standard', 'attention', 'hierarchical'
    'gate_hidden_ratio': 0.25,
    'gate_init_bias': -2.2,
    'gate_regularization_weight': 0.1,
    'gate_entropy_weight': 1.0,
    'gate_stability_weight': 0.01,
}


# Training loop modification example
def training_step_with_gating(model, batch, optimizer, step):
    """Example training step with gating losses."""
    # Forward pass
    carry = model.initial_carry(batch)
    new_carry, outputs = model(carry, batch)
    
    # Main task loss
    task_loss = compute_task_loss(outputs, batch['labels'])
    
    # Extract gate values (would need hooks in practice)
    gate_values = extract_gate_values(model)
    
    # Compute gating losses
    gate_losses = compute_gating_losses(gate_values)
    
    # Combine losses
    total_loss = task_loss
    for loss_name, loss_value in gate_losses.items():
        weight = GATING_CONFIG_DEFAULTS.get(f'{loss_name}_weight', 0.1)
        total_loss += weight * loss_value
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'task_loss': task_loss.item(),
        **{k: v.item() for k, v in gate_losses.items()}
    }


if __name__ == "__main__":
    # Quick test of gating network
    hidden_size = 256
    batch_size = 4
    seq_len = 32
    
    # Create gating network
    gate_net = HRMGatingNetwork(hidden_size)
    
    # Test inputs
    z_L = torch.randn(batch_size, seq_len, hidden_size)
    z_H = torch.randn(batch_size, seq_len, hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Compute gates
    g_L, g_H, g_X = gate_net(z_L, z_H, x)
    
    print(f"Gate shapes: L={g_L.shape}, H={g_H.shape}, X={g_X.shape}")
    print(f"Gate means: L={g_L.mean():.3f}, H={g_H.mean():.3f}, X={g_X.mean():.3f}")
    print(f"Gate ranges: L=[{g_L.min():.3f}, {g_L.max():.3f}]") 