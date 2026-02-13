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
from comlrl.trainers.actor_critic import MAACConfig, MAACTrainer
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


def aux_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 1) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    prompt_text = f"""Create a helper function for this coding problem.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Create a helper function named 'aux' that can assist the main function
- The function should return useful data for solving the problem

Your output should follow this format:

def aux(...):\n # your function code here\nreturn result\n"""

    return prompt_text


def main_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the main function generator (Agent 2) for code tasks."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux(...)

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function
- Implement ONLY the '{entry_point}' function as specified
- You can call aux() to assign value to a variable within your function if helpful

Your output should follow this format:

def {entry_point}({params_str}):\n # your function code here\nreturn result\n"""

    return prompt_text


def build_prompt_formatters() -> List:
    return [aux_function_formatter, main_function_formatter]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_prompt_reward_fn():
    def _reward(
        aux_outputs: List[str],
        main_outputs: List[str],
        *,
        batch_items=None,
    ) -> List[float]:
        count = min(len(aux_outputs), len(main_outputs))
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

        return execution_reward_aux(
            aux_outputs[:count],
            main_outputs[:count],
            test_cases,
            entry_points,
            raw_prompts,
        )

    return _reward


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-turn MAAC (shared critic) training for cooperative code generation."
    )
    add_config_args(parser)
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "maac_che_config.yaml"
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

    # Try to infer dataset type if missing
    if dataset_type is None and dataset_name:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
    if dataset_type is None:
        raise ValueError("dataset.type must be specified or inferrable from dataset.name")

    maac_cfg = config.get_section("maac") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", maac_cfg.get("seed", 42)))
    num_agents = maac_cfg.get("num_agents", 2)
    agent_names = config.get("agents")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(
            isinstance(x, str) for x in agent_names
        ):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(x) for x in agent_names]

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"maac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "config.yaml")

    _set_seed(seed_value)

    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    if agent_names:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in agent_names]
    else:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer_source)]
    for tok in tokenizers:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    tokenizer = tokenizers[0]

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
        display_model = (agent_names[0] if agent_names else model_name) or ""
        print(f"Using model: {display_model}")
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
    formatters = build_prompt_formatters()
    reward_fn = make_prompt_reward_fn()

    reward_processor = None
    shift_val = maac_cfg.get("reward_shift", -4)
    if shift_val is not None:
        try:
            shift_val_f = float(shift_val)
        except (TypeError, ValueError):
            shift_val_f = None
        if shift_val_f is not None:
            reward_processor = RewardProcessors.shift(value=shift_val_f)

    top_k = maac_cfg.get("top_k")
    temperature = maac_cfg.get("temperature", 0.6)
    top_p = maac_cfg.get("top_p", 0.6)
    model_kwargs: Dict[str, Any] = {}
    if model_config.torch_dtype is not None:
        model_kwargs["torch_dtype"] = model_config.torch_dtype
    critics_field = config.get("critics")
    critic_names = None
    if critics_field is not None:
        if not isinstance(critics_field, (list, tuple)) or not all(
            isinstance(x, str) for x in critics_field
        ):
            raise ValueError("critics must be a list of model names.")
        critic_names = [str(x) for x in critics_field]
    critic_config = config.get_critic_model_config(required=False)
    critic_name = critic_config.name if critic_config is not None else None
    critics = critic_names
    critic_model_kwargs: Dict[str, Any] = dict(model_kwargs)
    if critic_config is not None and critic_config.torch_dtype is not None:
        critic_model_kwargs["torch_dtype"] = critic_config.torch_dtype
    num_turns = maac_cfg.get("num_turns", 2)
    discount = maac_cfg.get("discount", 0.9)

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
            return get_external_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=num_agents,
                expert_model=expert_model,
                mode=external_mode,
                prompt_history_per_agent=prompt_history_per_agent,
                response_history_per_agent=response_history_per_agent,
            )

    model_arg = model_name or None
    agents_arg = agent_names
    trainer = MAACTrainer(
        agent_model=model_arg,
        agents=agents_arg,
        tokenizer=tokenizers if agent_names else tokenizer,
        reward_func=reward_fn,
        reward_processor=reward_processor,
        formatters=formatters,
        metrics_callback=None,
        external_transition=external_transition_fn,
        args=MAACConfig(
            num_turns=num_turns,
            num_train_epochs=maac_cfg.get("num_train_epochs", 40),
            agent_learning_rate=maac_cfg.get("agent_learning_rate", 5e-6),
            critic_learning_rate=maac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=maac_cfg.get("value_loss_coef", 0.6),
            rollout_buffer_size=maac_cfg.get("rollout_buffer_size", 8),
            max_new_tokens=maac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_agents=num_agents,
            num_generations=maac_cfg.get("num_generations", 1),
            discount=discount,
            critic_type=maac_cfg.get("critic_type", "v"),
            early_termination_threshold=maac_cfg.get(
                "early_termination_threshold", -0.2
            ),
            eval_interval=maac_cfg.get("eval_interval", 16),
            eval_num_samples=maac_cfg.get("eval_num_samples", 4),
            eval_batch_size=maac_cfg.get("eval_batch_size", 1),
            logging_steps=maac_cfg.get("logging_steps", 1),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": critic_model_kwargs,
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
    maac_section = config.get_section("maac") if hasattr(config, "get_section") else {}
    output_section = (
        config.get_section("output") if hasattr(config, "get_section") else {}
    )
    tags = wandb_section.get("tags", ["maac", dataset_name or "code", "turns_2"])
    wandb_name = (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or f"{(dataset_type or dataset_name)}-maac"
    )
    return {
        "project": wandb_section.get("project", "maac"),
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
                "num_turns": maac_section.get("num_turns", 2),
                "max_new_tokens": maac_section.get("max_new_tokens", 256),
                "temperature": maac_section.get("temperature", 0.6),
                "top_p": maac_section.get("top_p", 0.6),
                "top_k": maac_section.get("top_k"),
                "discount": maac_section.get("discount", 0.9),
                "critic_type": maac_section.get("critic_type", "v"),
            },
        },
    }


if __name__ == "__main__":
    main()
