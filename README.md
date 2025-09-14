# LLM Collaboration with MARL

This repository contains training scripts and configurations for the paper "LLM Collaboration with Multi‑Agent Reinforcement Learning".

## Datasets

- HumanEval (HE): 164 problems on split `test`
- CoopHumanEval (CHE): 82 problems on split `test`

## Usage

### Training by Default

```bash
# Single-agent HumanEval (GRPO)
python LLM_Collaboration_with_MARL/train_grpo.py \
  --config LLM_Collaboration_with_MARL/configs/grpo_he_config.yaml

# Multi-agent CoopHumanEval (MAGRPO)
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_che_config.yaml

# Multi-turn HumanEval (MT-MAGRPO)
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml
```

### With Parameter Overrides

You can override any configuration parameter using `--override`:

```bash
# Change model
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_he_config.yaml \
  --override model_name='bigcode/starcoder2-3b'

# Modify training params
python LLM_Collaboration_with_MARL/train_grpo.py \
  --config LLM_Collaboration_with_MARL/configs/grpo_che_config.yaml \
  --override grpo.num_train_epochs=20 grpo.learning_rate=3e-5

# Multi-turn override example
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_che_config.yaml \
  --override dataset.train_split='test[:20]' dataset.eval_split='test[20:30]' \
  magrpo.num_turns=2 magrpo.turn_gradient_weights=[1.5,0.5]
```
### Legacy Command-Line Arguments

You can also override with direct flags:

```bash
# Override model name
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_he_config.yaml \
  --model_name Qwen/Qwen2.5-Coder-7B

# Multi-turn direct args
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --num_epochs 10 --num_turns 2 --turn_gradient_weights 1.2 0.8
```

## Multi-Turn External Modes

Multi-turn training supports external transition modes for 2nd+ turns. Set via `magrpo.external_mode`:

- `expert_edits` (default): Uses an expert LLM to suggest edits per agent.
  - Requires `magrpo.expert_model` in config (e.g., `deepseek-coder`, Claude, etc.).
  - Requires corrsponding API keys in env vars.
- `level_passed`: Rule-based signals (impl found, syntax, tests summary, aux usage). No LLM needed.
- `level_feedback`: Detailed diagnostics (impl found, syntax with line/col, per-test pass/fail errors, aux usage). No LLM needed.
- `passed`: Minimal signal — only tells whether all sandbox tests passed. No LLM needed.

Run with a specific mode using overrides (no config edits required):

```bash
# HumanEval with detailed feedback signals
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override magrpo.num_turns=2 magrpo.external_mode='level_feedback'

# CoopHumanEval with concise signals
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_che_config.yaml \
  --override magrpo.num_turns=2 magrpo.external_mode='level_passed'
```

### Sandbox Tests vs Eval Tests

The external modes obtain `entry_point` and tests via an internal resolver registered by the training script. By default, the sandbox tests are the same as the dataset’s eval tests.
Note: `magrpo.sandbox_slice` only affects analysis-based modes (`level_feedback`, `level_passed`, `passed`). It has no effect on `expert_edits`.

- Use the full eval tests as sandbox (default): no action needed.
- Use only the first or last N eval tests as sandbox: set an integer slice in `magrpo`:

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override magrpo.num_turns=2 magrpo.external_mode='level_feedback' magrpo.sandbox_slice=1
```
