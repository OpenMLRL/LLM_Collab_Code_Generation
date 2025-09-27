# LLM Collaboration with MARL

Training scripts and configs for _"LLM Collaboration with Multi‑Agent Reinforcement Learning"_.

## Benchmarks

- MBPP: 427 problems on split `sanitized`
- HumanEval: 164 problems on split `test`
- CoopHumanEval: 82 problems on split `test`

## Training Scripts

```bash
python LLM_Collaboration_with_MARL/train_grpo.py \
  --config LLM_Collaboration_with_MARL/configs/grpo_he_config.yaml

python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_che_config.yaml
```

You can always override any configuration parameter using `--override`:

```bash
python LLM_Collaboration_with_MARL/train_magrpo.py \
  --config LLM_Collaboration_with_MARL/configs/magrpo_he_config.yaml \
  --override model.name='bigcode/starcoder2-3b' magrpo.num_turns=1
```

## Settings

### Joint Action Modes

`magrpo.joint_mode` determine how to combine each agent's K generations into joint actions at each turn. 2 modes are supported: if set 'align' by default, each agent's k-th generation is paired with the other agents' k-th generations to form a joint action; if set 'cross', all combinations of the agents' K generations are used to form joint actions (K^N joint actions for N agents).

Since the number of samples will also grow exponentially with the number of turns, aligned joint will be **more flexible** (\#samples could not be a perfect power) and hence faster to train in wall time. However, using cross joint will be **more sample efficient** (much lower VRAM compare to 'align' when num_generations=K^N), it also performs better since the **value estimation is more accurate**.

### Number of Turns

`magrpo.num_turns` determines the number of turns (`magrpo.num_turns=2` by default). The number of samples at each turn will grow exponentially with the number of turns: K^TN at turn T if cross joint, K^N if aligned joint. 

### Early Termination

`magrpo.termination_threshold` is used to incentive agents to find high-reward solutions quickly, instead of expanding the full Monte Carlo tree.

At each node (branch, turn), compute the mean immediate **reward across the sibling** joint actions at that node (siblings are the competing samples drawn at the same prompt/
  context/turn). If the mean exceeds the threshold, that branch stops expanding at this turn; training backpropagates from the truncated subtree. Other branches continue.

### Multi-Turn Prompt

`external.original_prompt` and `external.previous_response` both default as `true`. 2+ turn prompts include both the original first‑turn problem prompt and the previous response by default to preserve full context; you can shorten the context by setting either to `false` (for example, keep only the previous response to reduce tokens while retaining the most recent interaction).

### External Modes

`external.mode` is set to be 'level_feedback' by default. This gives additional information from external to prompts in the following turns; 'level_feedback' attaches test‑driven diagnostics, while alternatives include 'expert_edits' (an LLM proposes edits), 'level_passed'/'passed' (binary outcomes), and 'plain' (no signals). 

Specific settings for 'level_feedback' is `external.sandbox_slice`, which controls how many eval tests to include in the feedback. By default, sandbox executes only the first assert (sandbox_slice=1). Use all eval tests by setting `external.sandbox_slice` to 0, None, or 'all'. Negative values use the last asserts. `external.sandbox_slice` only affects analysis-based modes ('level_feedback', 'level_passed', 'passed'), and it has no effect on 'expert_edits'.

Specific settings for 'expert_edits' is `external.expert_edits_model`, which controls which LLM to use for proposing edits. By default, it uses DeepSeek-Coder. You can also change it to Claude-3, GPT-4, once you have keys/tokens in your global environment variables.

### Output

`output.save_model` is set to `false` by default because of the huge storage required by multiple LLMs. `verbose` is used for debug printing on cluster if set to be true, but it is default to be false and you can only see a tqdm bar that shows the training progress. You can also turn on `magrpo.log_code_levels` to log the level-rewards during training, but it will crazily slow down the training.