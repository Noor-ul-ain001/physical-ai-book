# Training Scripts

This directory contains scripts for training humanoid robots with reinforcement learning.

## Available Scripts

- `train_humanoid_rl.py`: Main training script using SAC algorithm
- `evaluate_policy.py`: Script for evaluating trained policies
- `collect_demonstrations.py`: Scripts for collecting expert demonstrations
- `transfer_learning.py`: Transfer learning utilities

## Usage

```bash
# Train a new policy
python train_humanoid_rl.py --config configs/humanoid_training_config.json

# Evaluate a trained policy
python evaluate_policy.py --model-path models/final_policy.pth
```
