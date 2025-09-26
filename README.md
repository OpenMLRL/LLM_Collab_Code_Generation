# LLM Collaboration with MARL

Training scripts and configs for _"LLM Collaboration with Multi‑Agent Reinforcement Learning"_.

## Benchmarks

- MBPP: 427 problems on split `sanitized`
- HumanEval: 164 problems on split `test`
- CoopHumanEval: 82 problems on split `test`

## Training Scripts

### Default Configs

```bash
python LLM_Collaboration_with_MARL/train_grpo.py \
  --config LLM_Collaboration_with_MARL/configs/grpo_he_config.yaml

python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_che_config.yaml
```

### Parameter Overrides

You can always override any configuration parameter using `--override`:

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_he_config.yaml \
  --override model.name='bigcode/starcoder2-3b' magrpo.num_turns=1
```
## Multi-Turn Settings

### Joint Actions & Early Termination

- Joint actions (magrpo.joint_mode):
  - `cross` (default): Form joint actions by Cartesian product of each agent's K generations (reuses sequences; no extra generation).
  - `aligned`: Join index‑aligned generations.

- Early termination (magrpo.termination_threshold / grpo.termination_threshold):
  - At each node (branch, turn), compute the mean immediate reward across the sibling joint actions at that node.
  - If the mean exceeds the threshold, that branch stops expanding at this turn; training backpropagates from the truncated subtree. Other branches continue.

Illustrative example (threshold = -0.2, 2 agents, K=2 → 4 joint actions per node):

```
Turn 1 (root) (a,b,c,d): [-1.5, -1.5, -1.0, -1.0]
mean rewards = -1.25 ≤ -0.2 → continue expanding all branches

  a (-1.5)
    ↳ Turn 2 children (e,f,g,h): [-1, -1, -1, -1]
      mean rewards = -1.0 ≤ -0.2 → continue

  b (-1.5)
    ↳ Turn 2 children (i,j,k,l): [0.0, 0.0, 0.0, -0.2]
      mean rewards = -0.05 > -0.2 → TERMINATE branch b here (no further children)

  c (-1.0)
    ↳ Turn 2 children: [...]

  d (-1.0)
    ↳ Turn 2 children: [...]
```

Notes:
- Termination is per‑branch; other branches continue normally.
- The same rule applies at deeper turns.
- For GRPO (single agent), the same threshold logic applies (one agent → one set of siblings per node).

### 2+Turn Prompt

By default, multi-turn prompts include both the original first‑turn problem prompt and the previous response.

- external.original_prompt: true (default)
- external.previous_response: true (default)

To exclude the original prompt but keep the previous response (shorter context):

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_he_config.yaml \
  --override external.original_prompt=False external.previous_response=True
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

#### Sandbox Tests

The external modes obtain `entry_point` and tests via an internal resolver registered by the training script. **By default, sandbox executes only the first assert (`sandbox_slice=1`).** Use all eval tests by setting `external.sandbox_slice` to `0`, `None`, or `'all'`. A negative value uses the last N asserts. Note: `external.sandbox_slice` only affects analysis-based modes (`level_feedback`, `level_passed`, `passed`), and it has no effect on `expert_edits`.

```bash
# Add an external.sandbox_slice override
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/mt_magrpo_che_config.yaml \
  --override external.mode='level_feedback' external.sandbox_slice=-2
```
