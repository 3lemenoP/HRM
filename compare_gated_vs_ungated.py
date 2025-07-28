#!/usr/bin/env python3
"""
Comparison script for gated vs ungated HRM models.
This script runs both configurations with identical hyperparameters for fair comparison.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_experiment(use_gating: bool, run_suffix: str):
    """Run a training experiment with or without gating."""
    
    # Base configuration - identical for both
    base_config = [
        "python", "pretrain.py",
        "data_path=data/arc-aug-1000",  # Update this path as needed
        "global_batch_size=128",
        "epochs=10000",
        "eval_interval=1000",
        "lr=1e-4",
        "lr_warmup_steps=500",
        "checkpoint_every_eval=true",
        "project_name=HRM-Gating-Comparison",
        
        # Small model for faster comparison
        "arch.hidden_size=256",
        "arch.H_layers=3",
        "arch.L_layers=3",
        "arch.H_cycles=2",
        "arch.L_cycles=3",
        "arch.halt_max_steps=3",
    ]
    
    # Add gating-specific configuration
    if use_gating:
        config = base_config + [
            "arch=hrm_v1_gated",  # Use gated architecture
            f"run_name=gated-{run_suffix}",
        ]
        print("\n" + "="*60)
        print(f"Starting GATED model training - {run_suffix}")
        print("="*60)
    else:
        config = base_config + [
            "arch=hrm_v1",  # Use standard architecture
            f"run_name=ungated-{run_suffix}",
        ]
        print("\n" + "="*60)
        print(f"Starting UNGATED model training - {run_suffix}")
        print("="*60)
    
    # Run the training
    start_time = time.time()
    result = subprocess.run(config, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")
    return True


def main():
    """Run comparison experiments."""
    
    print("HRM Gating vs Non-Gating Comparison")
    print("===================================")
    print("This script will train two models with identical hyperparameters:")
    print("1. Standard HRM (without gating)")
    print("2. HRM with adaptive gating mechanism")
    print("\nResults will be logged to Weights & Biases for comparison.")
    
    # Generate unique run suffix based on timestamp
    run_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Option to run sequentially or just show commands
    if len(sys.argv) > 1 and sys.argv[1] == "--show-commands":
        print("\nCommands to run manually:")
        print("\n# Ungated model:")
        print("python pretrain.py arch=hrm_v1 " + 
              f"run_name=ungated-{run_suffix} " +
              "global_batch_size=128 epochs=10000 ...")
        
        print("\n# Gated model:")
        print("python pretrain.py arch=hrm_v1_gated " + 
              f"run_name=gated-{run_suffix} " +
              "global_batch_size=128 epochs=10000 ...")
        return
    
    # Run experiments
    print("\nStarting experiments...")
    
    # Run ungated baseline first
    if not run_experiment(use_gating=False, run_suffix=run_suffix):
        print("Ungated experiment failed!")
        return 1
    
    # Run gated model
    if not run_experiment(use_gating=True, run_suffix=run_suffix):
        print("Gated experiment failed!")
        return 1
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nCheck Weights & Biases project 'HRM-Gating-Comparison' to compare:")
    print(f"- ungated-{run_suffix}")
    print(f"- gated-{run_suffix}")
    print("\nKey metrics to compare:")
    print("- Final accuracy (train/accuracy)")
    print("- Convergence speed")
    print("- Loss curves")
    print("- ACT steps used (train/steps)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 