#!/bin/bash
# Quick test script for HRM with gating mechanism
# This runs a small model for rapid testing

echo "======================================"
echo "HRM Gating Quick Test"
echo "======================================"

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: No data directory found."
    echo "Please prepare your dataset in the 'data' folder."
    exit 1
fi

# Run a small gated model for quick testing
echo "Starting training with small gated model..."
echo "Configuration:"
echo "- Architecture: HRM v1 with gating"
echo "- Hidden size: 128 (small for testing)"
echo "- Layers: 2 H-layers, 2 L-layers"
echo "- Batch size: 32"
echo "- Training steps: 1000"
echo ""

python pretrain.py \
    arch=hrm_v1_gated \
    arch.hidden_size=128 \
    arch.H_layers=2 \
    arch.L_layers=2 \
    arch.H_cycles=2 \
    arch.L_cycles=2 \
    arch.halt_max_steps=2 \
    global_batch_size=32 \
    epochs=1000 \
    eval_interval=200 \
    lr=1e-4 \
    lr_warmup_steps=100 \
    checkpoint_every_eval=false \
    project_name="HRM-Gating-QuickTest" \
    run_name="quick-test-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "======================================"
echo "Test completed!"
echo "Check Weights & Biases for results."
echo "======================================" 