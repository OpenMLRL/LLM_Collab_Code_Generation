"""
Unified training script for MAGRPO that supports both single-turn and multi-turn training.
The training mode is determined by the num_turns parameter in the config file.
Supports multiple datasets and configurations via YAML files.
"""

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from typing import Any, Dict

from config import Config, add_config_args, make_run_id, parse_overrides
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import make_code_reward_function
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
import external as external_ctx
from external import get_external_transition


def extract_function_params_from_prompt(prompt_text):
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def get_formatters(dataset_type: str, num_agents: int):
    """Get a list of per-agent formatters based on dataset type and agent count.

    For code tasks, use aux formatters for all agents except the last, which uses main.
    """
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"] and num_agents == 2:
        return [aux_function_formatter, main_function_formatter]

    raise NotImplementedError("Other number of agents have not been implemented yet")


def get_logger_and_aggregator(dataset_type: str, is_multi_turn: bool = False):
    """
    Get the logger and aggregator functions based on dataset type.
    Standardized to a single modern interface that accepts agent_completions_turns.
    """
    if dataset_type is None:
        return None, None

    # Use unified multi-turn compatible logger/aggregator for code datasets
    if dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def main():
    """Main function to run the unified MAGRPO training."""
    parser = argparse.ArgumentParser(
        description="Train MAGRPO with configurable dataset (single-turn or multi-turn)"
    )
    add_config_args(parser)

    

    args = parser.parse_args()
    if args.config:
        config = Config(args.config)
    else:
        raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    model_config = config.get_agent_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    output_base_dir = config.get("output.base_dir")

    # Try to infer dataset type from dataset name if not specified
    if dataset_type is None:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        elif "mbpp" in dataset_name.lower():
            dataset_type = "mbpp"
        else:
            raise ValueError(
                f"Could not infer dataset type from dataset name '{dataset_name}'. Please specify 'type' in dataset config."
            )
        print(f"Dataset type not specified, inferred as: {dataset_type}")

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")
    magrpo_config = (
        config.get_section("magrpo") if hasattr(config, "get_section") else {}
    )
    seed_value = int(config.get("seed", magrpo_config.get("seed", 42)))
    num_turns = magrpo_config.get("num_turns", 2)
    num_agents = magrpo_config.get("num_agents", 2)
    agent_names = config.get("agents")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(
            isinstance(x, str) for x in agent_names
        ):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(x) for x in agent_names]
    is_multi_turn = num_turns > 1
    output_verbose = bool(config.get("output.verbose", False))
    if output_verbose:
        print(f"Multi-turn training enabled: num_turns={num_turns}") if is_multi_turn else print(
            f"Single-turn training: num_turns={num_turns}"
        )

    run_id = make_run_id()

    if is_multi_turn:
        output_dir = os.path.join(output_base_dir, f"mt_job_{run_id}")
    else:
        output_dir = os.path.join(output_base_dir, f"job_{run_id}")

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)

    _set_seed(seed_value)

    train_dataset = None
    eval_dataset = None
    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if output_verbose:
        display_model = (agent_names[0] if agent_names else model_name) or ""
        print(f"\nUsing model: {display_model}")
        print(f"Model type: {model_config.type}")
        print(f"Max context window: {model_config.max_length} tokens")

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
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tok.padding_side = padding_side
        if model_config.special_tokens:
            if output_verbose:
                print("Adding special tokens...")
            tok.add_special_tokens(model_config.special_tokens)
            if output_verbose:
                print(
                    f"Special tokens added: {model_config.special_tokens.get('additional_special_tokens', [])}"
                )
    tokenizer = tokenizers[0]

    temperature = model_config.temperature
    top_p = model_config.top_p
    top_k = model_config.top_k
    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}
    _ext_passthrough = external_cfg.get("external_prompt_passthrough", False)
    if isinstance(_ext_passthrough, str):
        external_prompt_passthrough = _ext_passthrough.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    else:
        external_prompt_passthrough = bool(_ext_passthrough)

    # Register external context resolver using dataset items
    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    context_map = {}

    # Optionally restrict sandbox tests to the first N eval asserts
    # Default: keep only the first assert (sandbox_slice=1)
    # Set external.sandbox_slice to an integer N (>0) to keep the first N asserts,
    # or to 0 / None / 'all' to keep all eval asserts.
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

        # n > 0: keep first n asserts; n < 0: keep last |n| asserts; n == 0: keep all
        if n is None or n == 0:
            return test_code

        lines = test_code.splitlines()
        # Collect import preamble before check definition
        preamble = []
        check_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"\s*def\s+check\s*\(candidate\)\s*:\s*", line):
                check_idx = idx
                break
            preamble.append(line)

        # Find assert statements containing 'candidate'
        asserts = []
        search_start = check_idx + 1 if check_idx is not None else 0
        for line in lines[search_start:]:
            s = line.strip()
            if s.startswith("assert") and "candidate" in s:
                asserts.append(s)

        if not asserts:
            return test_code  # fallback when no asserts found

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
    if "train_dataset" in locals() and train_dataset is not None:
        _register_split(train_dataset)
    if "eval_dataset" in locals() and eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    magrpo_args_kwargs = {
        "num_turns": num_turns,
        "num_train_epochs": magrpo_config.get("num_train_epochs", 8),
        "agent_learning_rate": magrpo_config.get("agent_learning_rate", 2e-5),
        "logging_steps": magrpo_config.get("logging_steps", 50),
        "num_generations": magrpo_config.get("num_generations", 4),
        "max_new_tokens": magrpo_config.get("max_new_tokens", 256),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    magrpo_args_kwargs.update(
        {
            "num_agents": num_agents,
            "parallel_training": str(
                magrpo_config.get("parallel_training", "none")
            ).strip().lower(),
            "agent_devices": magrpo_config.get("agent_devices", ["cuda:0"]),
            "discount": magrpo_config.get("discount", 0.9),
            "joint_mode": magrpo_config.get("joint_mode", "aligned"),
            "early_termination_threshold": magrpo_config.get(
                "early_termination_threshold", -0.2
            ),
            "rollout_buffer_size": magrpo_config.get("rollout_buffer_size", 4),
            "train_batch_size": magrpo_config.get("train_batch_size", 4),
            "advantage_normalization": magrpo_config.get(
                "advantage_normalization", True
            ),
            "eval_interval": magrpo_config.get("eval_interval", 4),
            "eval_num_samples": magrpo_config.get("eval_num_samples", 4),
            "eval_batch_size": magrpo_config.get("eval_batch_size", 1),
            "external_prompt_passthrough": external_prompt_passthrough,
        }
    )
    magrpo_args = MAGRPOConfig(**magrpo_args_kwargs)
    formatters = get_formatters(dataset_type, num_agents)
    reward_func = make_code_reward_function(dataset_type, num_agents, config)
    eval_logger, eval_aggregator = get_logger_and_aggregator(
        dataset_type, is_multi_turn
    )
    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    if is_multi_turn:
        default_name = f"{dataset_type}-magrpo"
    else:
        default_name = f"{dataset_type}-magrpo"
    wandb_name = (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or default_name
    )
    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["magrpo", dataset_type or "code", f"turns_{num_turns}"]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags
    if external_mode == "level_feedback":
        if "self-evolved" not in tags:
            tags.append("self-evolved")
    dataset_section = config.get_section("dataset") if hasattr(config, "get_section") else {}
    model_section = (
        config.get_section("agent_model") if hasattr(config, "get_section") else {}
    )
    output_section = config.get_section("output") if hasattr(config, "get_section") else {}

    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": f"{wandb_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": tags,
        # Provide full sections for the trainer to log cleanly
        "config_sections": {
            "dataset": dataset_section,
            "agent_model": model_section,
            "output": output_section,
            "external": external_cfg,
            "trainer": magrpo_config,
        },
    }

    # Propagate verbosity to reward/external modules
    import rewards.code_rewards as code_rewards
    code_rewards.VERBOSE = bool(output_verbose)
    import external as external_mod
    external_mod.VERBOSE = bool(output_verbose)
    reward_processor = None
    if config.get("reward_processor.enabled", True):
        scale_factor = config.get("reward_processor.scale_factor", 1.0)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
        shift_val = config.get("reward_processor.shift", None)
        if shift_val is not None:
            try:
                shift_val_f = float(shift_val)
            except (TypeError, ValueError):
                shift_val_f = None
            if shift_val_f is not None:
                shift_proc = RewardProcessors.shift(value=shift_val_f)
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()
    # Build trainer kwargs (grouped: model/data, reward/formatting, logging, args)
    model_arg = model_name or None
    agents_arg = agent_names
    trainer_kwargs = {
        "agent_model": model_arg,
        "agents": agents_arg,
        "num_agents": num_agents,
        "tokenizer": tokenizers if agent_names else tokenizer,
        "model_config": {
            "torch_dtype": model_config.torch_dtype,
            "special_tokens": model_config.special_tokens,
        },
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "reward_func": reward_func,
        "formatters": formatters,
        # Logging / eval / config
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
        "dataset_type": dataset_type,
        "args": magrpo_args,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if (
        is_multi_turn
        and dataset_type
        and dataset_type.lower() in ["humaneval", "coophumaneval", "mbpp"]
    ):
        expert_model = external_cfg.get("expert_model", "deepseek-coder")

        def external_transition_wrapper(
            prompt,
            agent_completions,
            num_agents,
            prompt_history_per_agent=None,
            response_history_per_agent=None,
        ):
            # Returns full next-turn prompts per agent (strings)
            return get_external_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=num_agents,
                expert_model=expert_model,
                mode=external_mode,
                prompt_history_per_agent=prompt_history_per_agent,
                response_history_per_agent=response_history_per_agent,
            )

        trainer_kwargs["external_transition"] = external_transition_wrapper

    trainer = MAGRPOTrainer(**trainer_kwargs)
    trainer.verbose = bool(output_verbose)
    trainer.train()
    save_final = config.get("output.save_final_model", False)
    if save_final:
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
