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
  --override dataset.train_split='test[16:]' dataset.eval_split='test[:16]' \
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

Multi-turn training supports external transition modes for 2nd+ turns, set via `external.mode`:

- `level_feedback` **(default)**: Detailed diagnostics (impl found, syntax with line/col, per-test pass/fail errors, aux usage).
 - Requires `external.expert_model` in config when using `expert_edits` (e.g., `deepseek-coder`, Claude, etc.). This parameter is ignored for other modes (`level_feedback`, `level_passed`, `passed`, `plain`).
- Requires corrsponding API keys in env vars.
- `level_passed`: Binary passed signals (impl found, syntax, tests summary, aux usage).
- `passed`: A binary signal — "All levels passed" or "Not all levels passed".
- `plain`: No signals or diagnostics.

```bash
# HumanEval with detailed feedback signals
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override external.mode='level_feedback'
```

### Sandbox Tests

The external modes obtain `entry_point` and tests via an internal resolver registered by the training script. **By default, sandbox executes only the first assert (`sandbox_slice=1`).** Use all eval tests by setting `external.sandbox_slice` to `0`, `None`, or `'all'`. A negative value uses the last N asserts. Note: `external.sandbox_slice` only affects analysis-based modes (`level_feedback`, `level_passed`, `passed`), and it has no effect on `expert_edits`.

```bash
# Add an external.sandbox_slice override
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_che_config.yaml \
  --override external.mode='level_feedback' external.sandbox_slice=-2
```

### Handoff Strategy

In MAGRPO/GRPO multi-turn training, we hand off one prior completion per agent to keep compute bounded. The trainer selects this per the `handoff` mode: **default `random`**, or `best`. Selection happens in the CoMLRL trainer; external modes simply format the next-turn prompts using the provided completions. Configure via `magrpo.handoff` or `grpo.handoff` in your config or `--override`.


```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override external.mode='plain' magrpo.handoff='best'
```
