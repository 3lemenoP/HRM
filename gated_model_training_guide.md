# HRM with Gating: Training Guide

This guide provides step-by-step instructions for training the Hierarchical Reasoning Model (HRM) with the new gating mechanism implemented in Phase 1.

## Prerequisites

### 1. Environment Setup

First, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages include:
- PyTorch
- adam-atan2 (custom optimizer)
- einops
- hydra-core
- wandb (for experiment tracking)
- Flash Attention (optional but recommended)

### 2. Data Preparation

The model expects data in a specific format. You'll need to prepare your dataset according to the PuzzleDataset format. The default configuration expects data at `data/arc-aug-1000`.

## Configuration Setup

### Method 1: Using Pre-configured Gated Architecture

The easiest way is to use the provided gated architecture config:

```bash
python pretrain.py arch=hrm_v1_gated
```

### Method 2: Custom Configuration

Create your own config file or modify on the command line:

```bash
python pretrain.py \
    arch=hrm_v1 \
    arch.use_gating=true \
    arch.gate_hidden_ratio=0.25 \
    arch.gate_init_bias=-2.2
```

### Method 3: Create a Custom Config File

Create `config/arch/my_gated_config.yaml`:

```yaml
# Custom gated HRM configuration
defaults:
  - hrm_v1_gated

# Override specific parameters
hidden_size: 256  # Smaller model for testing
H_layers: 2
L_layers: 2

# Adjust gating parameters
gate_hidden_ratio: 0.3  # Slightly larger gate networks
gate_init_bias: -1.5    # Gates start more open
```

Then train with:
```bash
python pretrain.py arch=my_gated_config
```

## Training Commands

### Basic Training

Single GPU training with gating enabled:

```bash
python pretrain.py \
    arch=hrm_v1_gated \
    data_path=path/to/your/data \
    global_batch_size=128 \
    lr=1e-4
```

### Distributed Training

Multi-GPU training with PyTorch distributed:

```bash
torchrun --nproc_per_node=4 pretrain.py \
    arch=hrm_v1_gated \
    data_path=path/to/your/data \
    global_batch_size=512 \
    lr=2e-4
```

### Training with Custom Parameters

```bash
python pretrain.py \
    arch=hrm_v1_gated \
    data_path=path/to/your/data \
    global_batch_size=256 \
    epochs=50000 \
    eval_interval=5000 \
    lr=1e-4 \
    lr_warmup_steps=1000 \
    arch.gate_init_bias=-2.5 \
    arch.halt_max_steps=3 \
    project_name="HRM-Gating-Experiments" \
    run_name="gated-baseline"
```

## Hyperparameter Tuning Guide

### 1. Gating-Specific Parameters

**`use_gating`** (bool, default: false)
- Set to `true` to enable gating mechanism

**`gate_hidden_ratio`** (float, default: 0.25)
- Controls the size of gating networks
- Lower values (0.1-0.2): More efficient, less expressive
- Higher values (0.3-0.5): More expressive, higher compute cost
- Recommended: Start with 0.25

**`gate_init_bias`** (float, default: -2.2)
- Controls initial gate openness
- sigmoid(-2.2) ≈ 0.1 (10% open)
- More negative (-3.0): Gates start nearly closed
- Less negative (-1.0): Gates start more open (sigmoid(-1.0) ≈ 0.27)
- Recommended: Start with -2.2, adjust based on task

### 2. General Training Parameters

**`global_batch_size`**
- Total batch size across all GPUs
- Larger batches generally improve gating stability
- Recommended: 256-1024 depending on GPU memory

**`lr`** (learning rate)
- May need slight adjustment with gating
- If gates saturate quickly, try lower LR (5e-5)
- If gates don't learn, try higher LR (2e-4)

**`lr_warmup_steps`**
- Important for gating stability
- Recommended: 1000-2000 steps

### 3. Architecture Parameters

**`H_cycles` / `L_cycles`**
- Number of update cycles for H and L modules
- With gating, you might achieve same performance with fewer cycles
- Try reducing by 1-2 cycles after gating works well

**`halt_max_steps`**
- Maximum ACT steps
- Gating may allow more efficient computation, try reducing

## Monitoring Training

### 1. Weights & Biases (W&B) Integration

Training automatically logs to W&B. Key metrics to monitor:

**Gating-specific metrics** (need to be added in Phase 2):
- Gate activation means (should stay in 0.1-0.9 range)
- Gate saturation (% of gates < 0.1 or > 0.9)
- Module contribution ratios

**Standard metrics**:
- `train/lm_loss`: Language modeling loss
- `train/accuracy`: Token-level accuracy
- `train/exact_accuracy`: Sequence-level accuracy
- `train/steps`: ACT steps used

### 2. Checkpointing

Checkpoints are saved to `checkpoints/[project_name]/[run_name]/`

To resume training from a checkpoint:
```bash
python pretrain.py \
    arch=hrm_v1_gated \
    checkpoint_path=checkpoints/existing/run/path \
    # ... other parameters
```

## Debugging Common Issues

### 1. Gates Saturating (all 0 or 1)

**Symptoms**: Gates quickly go to extreme values
**Solutions**:
- Reduce learning rate
- Adjust `gate_init_bias` (try -1.5 or -3.0)
- Add gate regularization (Phase 2 feature)

### 2. No Gating Effect

**Symptoms**: Gated and ungated models perform identically
**Solutions**:
- Verify `use_gating=true` in config
- Check gate values aren't all uniform
- Increase `gate_hidden_ratio` for more expressive gates

### 3. Training Instability

**Symptoms**: Loss spikes or NaN values
**Solutions**:
- Increase warmup steps
- Reduce learning rate
- Start with `gate_init_bias=-3.0` (nearly closed gates)

### 4. Memory Issues

**Solutions**:
- Reduce `global_batch_size`
- Reduce `hidden_size` or number of layers
- Use gradient accumulation (not shown in basic config)

## Example Training Scenarios

### Scenario 1: Quick Testing

Test if gating works with a small model:

```bash
python pretrain.py \
    arch=hrm_v1_gated \
    arch.hidden_size=128 \
    arch.H_layers=2 \
    arch.L_layers=2 \
    global_batch_size=32 \
    epochs=1000 \
    eval_interval=100
```

### Scenario 2: Full Training Run

Production training with optimal settings:

```bash
torchrun --nproc_per_node=8 pretrain.py \
    arch=hrm_v1_gated \
    data_path=data/arc-full \
    global_batch_size=768 \
    epochs=100000 \
    eval_interval=10000 \
    lr=1e-4 \
    lr_warmup_steps=2000 \
    checkpoint_every_eval=true \
    project_name="HRM-Gating-Production" \
    run_name="gated-full-v1"
```

### Scenario 3: Ablation Study

Compare gated vs ungated:

```bash
# Ungated baseline
python pretrain.py \
    arch=hrm_v1 \
    run_name="ungated-baseline" \
    # ... other params

# Gated version
python pretrain.py \
    arch=hrm_v1_gated \
    run_name="gated-default" \
    # ... same other params
```

## Next Steps

1. **Run initial tests** with small models to verify gating works
2. **Monitor gate statistics** during training (Phase 2 will add visualizations)
3. **Compare performance** between gated and ungated models
4. **Experiment with gate parameters** for your specific task
5. **Share results** to help optimize default settings

## Additional Resources

- `config/arch/hrm_v1_gated.yaml` - Default gated architecture config
- `test_gating_phase1.py` - Test suite for gating functionality
- `gating_implementation_plan.md` - Full technical details
- `phase1_implementation_summary.md` - Implementation overview 