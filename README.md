# LLM Collaboration with MARL

This repository contains training scripts and configurations for paper _"LLM Collaboration with Multi-Agent Reinforcement Learning"_.

## Experiments

- **HumanEval**: Code generation benchmark
- **CoopHumanEval**: Cooperative code generation

## Usage

### Training by Default

You can run the training scripts with default configurations as follows:

```bash
# Single-agent HumanEval
python experiments/train_grpo.py --config experiments/configs/grpo_he_config.yaml

# Multi-agent CoopHumanEval
python experiments/train_magrpo.py --config experiments/configs/magrpo_che_config.yaml

# Multi-turn HumanEval
python experiments/train_magrpo.py --config experiments/configs/mt_magrpo_he_config.yaml
```

### With Parameter Overrides

You can override any configuration parameter using the `--override` flag:

```bash
# Change model
python experiments/train_magrpo.py --config experiments/configs/magrpo_he_config.yaml \
    --override model_name=bigcode/starcoder2-3b

# Modify training parameters
python experiments/train_grpo.py --config experiments/configs/grpo_che_config.yaml \
    --override grpo.num_train_epochs=20 grpo.learning_rate=5e-6

# Change dataset splits  
python experiments/train_magrpo.py --config experiments/configs/magrpo_che_config.yaml \
    --override dataset.train_split="train[:1000]" dataset.eval_split="test[:100]"

# Multiple overrides for multi-turn training
python experiments/train_magrpo.py --config experiments/configs/mt_magrpo_che_config.yaml \
    --override magrpo.num_turns=2 magrpo.turn_gradient_weights=[1.5,0.5]
```

### Legacy Command-Line Arguments

For backward compatibility, you can also use direct command-line arguments:

```bash
# Override model name
python experiments/train_magrpo.py --config experiments/configs/magrpo_he_config.yaml \
    --model_name Qwen/Qwen2.5-Coder-7B

# For multi-turn, override specific parameters
python experiments/train_magrpo.py --config experiments/configs/mt_magrpo_he_config.yaml \
    --num_epochs 10 --num_turns 2 --turn_gradient_weights 1.2 0.8
```
