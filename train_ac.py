import argparse
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.actor_critic import IACConfig, IACTrainer
from comlrl.utils.reward_processor import RewardProcessors
from rewards.code_rewards import execution_reward_aux
import external as external_ctx
from external import get_external_transition


def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def complete_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the complete function generator (single agent)."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Implement ONLY the '{entry_point}' function as specified
- Make sure your solution is complete and handles all cases

Your output should follow this format:

def {entry_point}({params_str}):\n    # your function code here\n    return result\n"""

    return prompt_text


def get_formatter(dataset_type: str):
    """Get the appropriate formatter based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval/mbpp' to the dataset section."
        )

    formatters_map = {
        "humaneval": complete_function_formatter,
        "coophumaneval": complete_function_formatter,
        "mbpp": complete_function_formatter,
    }
    return formatters_map.get(dataset_type.lower(), complete_function_formatter)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_prompt_reward_fn():
    def _reward(
        outputs: List[str],
        *,
        batch_items=None,
    ) -> List[float]:
        count = len(outputs)
        if count == 0:
            return []
        if batch_items:
            if len(batch_items) >= count:
                items = list(batch_items)[:count]
            else:
                items = [batch_items[0]] * count
            test_cases = [item.get("test", "") for item in items]
            entry_points = [item.get("entry_point", "") for item in items]
            raw_prompts = [item.get("prompt", "") for item in items]
        else:
            raise ValueError("batch_items must be provided for reward calculation")

        aux_outputs = [""] * count
        main_outputs = outputs[:count]

        return execution_reward_aux(
            aux_outputs,
            main_outputs,
            test_cases,
            entry_points,
            raw_prompts,
        )

    return _reward


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-agent Actor-Critic training for code generation."
    )
    add_config_args(parser)
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "ac_che_config.yaml"
        if default_config_path.exists():
            config = Config(str(default_config_path))
        else:
            raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    model_config = config.get_agent_model_config()
    critic_config = None
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    train_split = config.get("dataset.train_split") or config.get(
        "dataset.split", "train"
    )
    eval_split = config.get("dataset.eval_split")
    train_size = config.get("dataset.size")
    eval_size = config.get("dataset.eval_size")
    output_base_dir = config.get("output.base_dir", "output")
    output_verbose = bool(config.get("output.verbose", False))

    if dataset_type is None and dataset_name:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
    if dataset_type is None:
        raise ValueError("dataset.type must be specified or inferrable from dataset.name")

    # AC-specific config (needed early for seed)
    ac_cfg = config.get_section("ac") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", ac_cfg.get("seed", 42)))

    num_agents = int(ac_cfg.get("num_agents", 1))
    if num_agents != 1:
        raise ValueError("train_ac expects ac.num_agents=1. Use train_iac for multi-agent.")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"ac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "config.yaml")

    _set_seed(seed_value)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(dataset_name, split=train_split)
    if train_size is not None:
        train_size = min(int(train_size), len(train_dataset))
        train_dataset = train_dataset.select(range(train_size))
    else:
        train_size = len(train_dataset)

    eval_dataset = None
    if eval_split:
        eval_dataset = load_dataset(dataset_name, split=eval_split)
        if eval_size is not None:
            eval_size = min(int(eval_size), len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_size))
        else:
            eval_size = len(eval_dataset)

    if output_verbose:
        print(f"Using model: {model_name}")
        print(f"Train dataset: {dataset_name} split={train_split} size={train_size}")
        if eval_dataset is not None:
            print(f"Eval dataset: {dataset_name} split={eval_split} size={eval_size}")

    config.update(
        {
            "dataset": {
                "type": dataset_type,
                "train_split": train_split,
                "eval_split": eval_split,
                "size": train_size,
                "eval_size": eval_size,
            }
        }
    )
    if hasattr(config, "save"):
        config.save(config_save_path)

    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}

    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    context_map = {}
    _sandbox_val = external_cfg.get("sandbox_slice", 1)
    if isinstance(_sandbox_val, str):
        _sv = _sandbox_val.strip().lower()
        if _sv == "all":
            sandbox_slice = 0
        elif _sv.lstrip("-").isdigit():
            sandbox_slice = int(_sv)
        else:
            sandbox_slice = None
    elif isinstance(_sandbox_val, int):
        sandbox_slice = _sandbox_val
    else:
        sandbox_slice = None if _sandbox_val is None else 0

    def _make_sliced_assert_tests(test_code: str, n: int) -> str:
        if not isinstance(test_code, str) or not test_code.strip():
            return test_code
        if n is None or n == 0:
            return test_code
        lines = test_code.splitlines()
        preamble = []
        check_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"\s*def\s+check\s*\(candidate\)\s*:\s*", line):
                check_idx = idx
                break
            preamble.append(line)
        asserts = []
        search_start = check_idx + 1 if check_idx is not None else 0
        for line in lines[search_start:]:
            s = line.strip()
            if s.startswith("assert") and "candidate" in s:
                asserts.append(s)
        if not asserts:
            return test_code
        preamble_text = "\n".join(preamble).strip()
        new_parts = []
        if preamble_text:
            new_parts.append(preamble_text)
        new_parts.append("def check(candidate):")
        selected = asserts[:n] if n > 0 else asserts[n:]
        for a in selected:
            new_parts.append(f"    {a}")
        return "\n".join(new_parts) + "\n"

    def _register_split(ds):
        for item in ds:
            key = _normalize_prompt(item.get("prompt", ""))
            if key and key not in context_map:
                tests_eval = item.get("test", "")
                tests_sandbox = (
                    _make_sliced_assert_tests(tests_eval, sandbox_slice)
                    if sandbox_slice is not None and sandbox_slice != 0
                    else tests_eval
                )
                context_map[key] = {
                    "entry_point": item.get("entry_point", ""),
                    "tests_eval": tests_eval,
                    "tests_sandbox": tests_sandbox,
                }
    if train_dataset is not None:
        _register_split(train_dataset)
    if eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    # Propagate verbosity to reward/external modules
    import rewards.code_rewards as code_rewards

    code_rewards.VERBOSE = bool(output_verbose)
    import external as external_mod

    external_mod.VERBOSE = bool(output_verbose)
    formatter = get_formatter(dataset_type)
    reward_fn = make_prompt_reward_fn()

    reward_processor = None
    shift_val = ac_cfg.get("reward_shift", -4)
    if shift_val is not None:
        try:
            shift_val_f = float(shift_val)
        except (TypeError, ValueError):
            shift_val_f = None
        if shift_val_f is not None:
            reward_processor = RewardProcessors.shift(value=shift_val_f)

    # AC-specific config
    top_k = model_config.top_k
    temperature = model_config.temperature
    top_p = model_config.top_p
    use_separate_critic = bool(ac_cfg.get("use_separate_critic", True))
    model_kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        model_kwargs["torch_dtype"] = model_config.torch_dtype
    critic_names = None
    critics_field = config.get("critics")
    if critics_field is not None:
        if not isinstance(critics_field, (list, tuple)) or not all(
            isinstance(x, str) for x in critics_field
        ):
            raise ValueError("critics must be a list of model names.")
        critic_names = [str(x) for x in critics_field]
    critic_config = config.get_critic_model_config(required=False)
    critic_name = critic_config.name if critic_config is not None else None
    critics = critic_names
    critic_model_kwargs = dict(model_kwargs)
    if critic_config is not None and critic_config.torch_dtype is not None:
        critic_model_kwargs["torch_dtype"] = critic_config.torch_dtype
    num_turns = ac_cfg.get("num_turns", 1)
    rollout_buffer_size = ac_cfg.get("rollout_buffer_size", 8)

    external_transition_fn = None
    if num_turns > 1:
        external_mode = external_cfg.get("mode", "level_feedback")
        expert_model = external_cfg.get("expert_model", "deepseek-coder")

        def external_transition_fn(
            prompt,
            agent_completions,
            num_agents,
            prompt_history_per_agent=None,
            response_history_per_agent=None,
        ):
            prompts = get_external_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=num_agents,
                expert_model=expert_model,
                mode=external_mode,
                prompt_history_per_agent=prompt_history_per_agent,
                response_history_per_agent=response_history_per_agent,
            )
            if isinstance(prompts, (list, tuple)):
                return list(prompts)
            return [str(prompts)]

    trainer = IACTrainer(
        agent_model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        reward_processor=reward_processor,
        formatters=formatter,
        metrics_callback=None,
        external_transition=external_transition_fn,
        args=IACConfig(
            num_turns=num_turns,
            num_train_epochs=ac_cfg.get("num_train_epochs", 40),
            agent_learning_rate=ac_cfg.get("agent_learning_rate", 5e-6),
            critic_learning_rate=ac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=ac_cfg.get("value_loss_coef", 0.6),
            advantage_normalization=ac_cfg.get("advantage_normalization", True),
            value_clip_range=ac_cfg.get("value_clip_range", 0.2),
            rollout_buffer_size=rollout_buffer_size,
            max_new_tokens=ac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_agents=1,
            num_generations=ac_cfg.get("num_generations", 1),
            use_separate_critic=use_separate_critic,
            parallel_training=str(ac_cfg.get("parallel_training", "mp")).strip().lower(),
            agent_devices=ac_cfg.get("agent_devices", None),
            critic_devices=ac_cfg.get("critic_devices", None),
            critic_value_head_hidden_dim=ac_cfg.get("critic_value_head_hidden_dim"),
            value_head_hidden_dim=ac_cfg.get("value_head_hidden_dim"),
            discount=ac_cfg.get("discount", 0.9),
            early_termination_threshold=ac_cfg.get(
                "early_termination_threshold", -0.2
            ),
            eval_interval=ac_cfg.get("eval_interval", 16),
            eval_num_samples=ac_cfg.get("eval_num_samples", 4),
            eval_batch_size=ac_cfg.get("eval_batch_size", 1),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": (
                critic_model_kwargs
                if critic_config is not None
                else model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(
            config,
            dataset_name,
            dataset_type,
            train_split,
            eval_split,
            train_size,
            eval_size,
        ),
        critic_model=critic_name,
        critics=critics,
    )
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get("output.save_path", os.path.join(output_dir, "final_model"))
        trainer.save_model(save_path)
        if output_verbose:
            print(f"Model saved to: {save_path}")

    if wandb.run is not None:
        wandb.finish()


def _build_wandb_config(
    config: Config,
    dataset_name: str,
    dataset_type: str | None,
    train_split: str,
    eval_split: str,
    train_size: int,
    eval_size: int | None,
):
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    ac_section = config.get_section("ac") if hasattr(config, "get_section") else {}
    model_section = (
        config.get_section("agent_model") if hasattr(config, "get_section") else {}
    )
    output_section = (
        config.get_section("output") if hasattr(config, "get_section") else {}
    )
    tags = wandb_section.get("tags", ["ac", dataset_name or "code", "turns_1"])
    wandb_name = (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or f"{(dataset_type or dataset_name)}-ac"
    )
    return {
        "project": wandb_section.get("project", "ac"),
        "entity": wandb_section.get("entity"),
        "name": wandb_name,
        "dir": wandb_section.get("dir"),
        "tags": tags,
        "config_sections": {
            "dataset": {
                "name": dataset_name,
                "train_split": train_split,
                "eval_split": eval_split,
                "train_size": train_size,
                "eval_size": eval_size,
            },
            "output": output_section,
            "trainer": {
                "num_turns": ac_section.get("num_turns", 1),
                "max_new_tokens": ac_section.get("max_new_tokens", 256),
                "temperature": model_section.get("temperature"),
                "top_p": model_section.get("top_p"),
                "top_k": model_section.get("top_k"),
                "discount": ac_section.get("discount", 0.9),
                "use_separate_critic": ac_section.get("use_separate_critic", True),
            },
        },
    }


if __name__ == "__main__":
    main()
