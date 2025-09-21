# LLM Collaboration with MARL

This repository contains training scripts and configurations for the paper "LLM Collaboration with Multi‑Agent Reinforcement Learning".
- [Benchmarks](#benchmarks)
- [Training Scripts](#training-scripts)
  - [Default Configs](#default-configs)
  - [Parameter Overrides](#parameter-overrides)
  - [Legacy Command-Line Args](#legacy-command-line-args)
- [Multi-Turn Settings](#multi-turn-settings)
  - [2+Turn Prompt Composition](#2turn-prompt-composition)
  - [External Modes](#external-modes)
  - [Sandbox Tests](#sandbox-tests)

## Benchmarks

- HumanEval (HE): 164 problems on split `test`
- CoopHumanEval (CHE): 82 problems on split `test`

## Training Scripts

### Default Configs

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

### Parameter Overrides

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
### Legacy Command-Line Args

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
## Multi-Turn Settings

### 2+Turn Prompt Composition

To save memory usage, 2+ turn prompts **include the previous response without the original first‑turn problem prompt by default**. You can add the original prompt to match the concept of observation-action history in MARL.

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override magrpo.external_original_prompt=True magrpo.external_previous_response=True
```

### External Modes

Multi-turn training supports external transition modes for 2nd+ turns, set via `magrpo.external_mode`:

- `expert_edits` **(default)**: Uses an expert LLM to suggest edits.
  - Requires `magrpo.expert_model` in config (e.g., `deepseek-coder`, Claude, etc.).
  - Requires corrsponding API keys in env vars.
- `level_passed`: Binary passed signals (impl found, syntax, tests summary, aux usage).
- `level_feedback`: Detailed diagnostics (impl found, syntax with line/col, per-test pass/fail errors, aux usage).
- `passed`: A binary signal — "All levels passed" or "Not all levels passed".
- `plain`: No signals or diagnostics.

```bash
# HumanEval with detailed feedback signals
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override magrpo.external_mode='level_feedback'
```

### Sandbox Tests

The external modes obtain `entry_point` and tests via an internal resolver registered by the training script. **By default, the sandbox tests are the same as the dataset’s eval tests.**
Note: `magrpo.sandbox_slice` only affects analysis-based modes (`level_feedback`, `level_passed`, `passed`), and it has no effect on `expert_edits`.

```bash
# Add a magrpo.sandbox_slice to override
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_che_config.yaml \
  --override magrpo.external_mode='level_feedback' magrpo.sandbox_slice=-2
```

### Handoff Strategy

In MAGRPO, since agents generate a few responses per turn, we need to hand off one for efficiency, else the number of generations per turn will increase exponentially. External handoff controls which previous response is used as context for the later turns. **By default, the "best" completion per agent from the prior turn is used.** Random handoff requires the training loop to supply a candidate pool of previous-turn completions per agent to the external transition. If only a single completion per agent is available, random falls back to the best completion.


```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override magrpo.external_mode='plain' magrpo.external_handoff='random'
```
