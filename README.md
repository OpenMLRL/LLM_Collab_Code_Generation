# LLM Collaboration – Code Generation

Code-generation environments for [CoMLRL](https://github.com/OpenMLRL/CoMLRL), used in the code-generation experiments of [**[AAAI26]** **_LLM Collaboration with Multi‑Agent Reinforcement Learning_**](https://arxiv.org/abs/2508.04652). One agent writes an auxiliary function, the other writes the main function that uses it, and both are trained with a shared reward.

<img src="./demo_cg.gif" alt="Code generation demo" width="600px">

## Installation

CoMLRL can be installed from PyPI or conda-forge (`conda install -c conda-forge comlrl`). Users should also install a PyTorch build compatible with their device.

```bash
pip install comlrl
git clone https://github.com/OpenMLRL/LLM_Collab_Code_Generation.git
cd LLM_Collab_Code_Generation
```

A `CoMLRL/` checkout next to this repository takes precedence over the installed package, so local CoMLRL changes are used without reinstalling.

## Quick Start

The following command trains two agents with MAGRPO on CoopHumanEval:

```bash
python train_magrpo.py --config configs/magrpo_che_config.yaml
```

Any configuration value can be changed with `--override`:

```bash
python train_magrpo.py --config configs/magrpo_he_config.yaml \
  --override agent_model.name='bigcode/starcoder2-3b' magrpo.num_turns=1
```

## Tasks

- HumanEval (`he` configs): 164 problems on split `test`.
- CoopHumanEval (`che` configs): 82 problems on split `test`.

MAGRPO, IAC, MAAC, and the baselines use two turns by default, while the preference methods are single-turn. Each agent keeps its own history of prompts and responses and sees no transcript of the other agent, so the agents coordinate only through the shared reward. Between turns, `external.mode` builds the next prompt:

- `level_feedback` (default): syntax and test diagnostics, followed by a request for a code-only revision. `external.sandbox_slice` sets how many asserts are shown: 1 by default; 0, `None`, or `'all'` for every test; negative values count from the end.
- `expert_edits`: an external LLM (`external.expert_model`, DeepSeek-Coder by default) first suggests compact edits.
- `plain`: the previous attempt with a request for a concise, code-only revision.

The rollout tree is controlled by `magrpo.num_turns`, `magrpo.num_generations`, `magrpo.joint_mode` (`aligned` or `cross`), and `magrpo.early_termination_threshold`, as described in [Multi-Turn Interaction](https://openmlrl.github.io/CoMLRL/docs/marl/multi-turn-interaction/).

## Reward

`rewards/code_rewards.py` scores each auxiliary/main pair in three levels, and the same rewards back evaluation:

1. **Definitions:** +0.4 for a well-defined auxiliary function with a return value and +0.6 for a correct entry-point function. Scoring stops if the main function is missing.
2. **Syntax:** both functions are joined with the prompt's imports. A clean syntax check gives +0.5 and allows execution.
3. **Execution:** prompt-derived tests run in a sandbox (10 s per assert, at most three timeouts). Passing asserts earn up to +1.0, with +0.5 when the main function uses the auxiliary function, +1.0 when the main function is not a thin wrapper, and −0.5 when the auxiliary return value is ignored.

## Training Scripts

Configs are in `configs/`. Without `--config`, each script uses its CoopHumanEval config, except `train_grpo.py`, which uses HumanEval.

| Script | Method | Configs |
| --- | --- | --- |
| `train_magrpo.py` | MAGRPO | `he`, `che` |
| `train_iac.py`, `train_maac.py` | IAC, MAAC | `he`, `che` |
| `train_madpo.py`, `train_madpo_iter.py` | MADPO and its iterative variant | `che` |
| `train_marlhf.py`, `train_marlhf_iter.py` | MARLHF and its iterative variant | `che` |
| `train_grpo.py`, `train_ac.py` | Single-agent GRPO and actor-critic baselines | `he`, `che` |

MADPO and MARLHF build preferences from task rewards. MARLHF then trains a joint reward model and runs online RL with `rl_algorithm: magrpo` by default. The iterative variants refresh the preferences at each iteration against a comparator, as described in [Multi-LLM PbRL](https://openmlrl.github.io/CoMLRL/docs/marl/multi-llm-pbrl/).

## Options

**Centralized collaboration.** One model can play both roles with `--override <method>.collaboration_mode=centralized`, for example `magrpo.collaboration_mode=centralized`; the default is decentralized. `num_agents` stays 2 for the two roles, while only one model is loaded. The joint response contains `<auxiliary>` and `<main>` sections, which are split only for rewards and evaluation. `max_new_tokens` then limits the whole joint response, and MARLHF's `reward_max_length` has to fit the joint prompt and both outputs.

**Native centralized prompt (MAGRPO).** By default the centralized prompt concatenates the Auxiliary and Main agent prompts, so the problem appears twice and each copy asks for "only the function". `magrpo.centralized_prompt=native` states the problem once and asks one programmer for `aux` and the main function in the same `<auxiliary>` / `<main>` sections; parsing, reward and evaluation are unchanged. Centralized MAGRPO is single-turn, so the decentralized runs it is compared with use `magrpo.num_turns=1`, and the joint response gets the two roles' combined budget:

```bash
python train_magrpo.py --config configs/magrpo_che_config.yaml --override \
  magrpo.collaboration_mode=centralized magrpo.num_turns=1 \
  magrpo.max_new_tokens=512 magrpo.centralized_prompt=native
```

**Bounded centralized answers.** A free-running joint answer can drift: after a few updates the model may drop the section tags and keep writing prose after the code until `max_new_tokens`, and an answer cut off before its functions are complete scores zero. `magrpo.centralized_answer=bounded` ends either centralized prompt with the opening `<auxiliary>` tag and stops every train and eval rollout at the first `</main>`. The reward, parser and prompts are unchanged; the default is `open`.

**Checkpoints.** A top-level `checkpoint` section is passed to CoMLRL's `trainer.train()`; without it, no training checkpoints are saved.

```yaml
checkpoint:
  checkpoint_dir: ./checkpoints/code-run
  checkpoint_interval: 1
  checkpoint_total_limit: 2
  # resume_from_checkpoint: ./checkpoints/code-run/checkpoint-1
```

Resuming does not save new checkpoints unless `checkpoint_dir` is also set. A resumed run continues logging to the saved W&B run unless `wandb.id` or `wandb.resume` is set.

**Logging.** The loggers report pass rates, syntax success, reward components, termination statistics, and auxiliary-function usage. W&B is configured through `wandb.project`, `wandb.entity`, and `wandb.name`. `output.save_final_model` is `false` by default, and `output.verbose` prints detailed per-episode traces.
