"""
Unified training script for MAGRPO that supports both single-turn and multi-turn training.
The training mode is determined by the num_turns parameter in the config file.
Supports multiple datasets and configurations via YAML files.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Single-turn code logger no longer used directly; multi-turn logger handles all cases
from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_aux
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
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
    if dataset_type.lower() in ["humaneval", "coophumaneval"] and num_agents == 2:
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
    if dataset_type.lower() in ["humaneval", "coophumaneval"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    return None, None


def get_reward_function(dataset_type: str, num_agents: int):
    """Get a reward function compatible with variable number of agents (single-turn).

    For code tasks, map N-agent completions to the existing aux/main reward by
    using the first agent as aux and the last agent as main.
    """
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
        )

    if dataset_type.lower() in ["humaneval", "coophumaneval"]:

        def reward_wrapper(*agent_completions, batch_items=None, prompts=None):
            # agent_completions: tuple of lists (one list per agent), each list contains strings per completion
            if not agent_completions or len(agent_completions) < 1:
                return []

            # Choose aux from first agent if available when >=2, otherwise empty list
            if len(agent_completions) >= 2:
                completion1 = agent_completions[0]
                completion2 = agent_completions[-1]
            else:
                completion1 = [""] * len(agent_completions[0])
                completion2 = agent_completions[0]

            test_cases = []
            entry_points = []
            original_prompts = []

            if batch_items is not None:
                for item in batch_items:
                    test_cases.append(item["test"])
                    entry_points.append(item["entry_point"])
                    original_prompts.append(item.get("prompt", ""))
            else:
                raise ValueError("batch_items must be provided for reward calculation")

            return execution_reward_aux(
                completion1, completion2, test_cases, entry_points, original_prompts
            )

        return reward_wrapper

    raise ValueError(f"Unknown dataset type: {dataset_type}")


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

    # Apply command-line overrides
    

    # Load model configuration
    model_config = config.get_model_config()
    model_name = model_config.name
    output_base_dir = config.get("output.base_dir")
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")

    # Try to infer dataset type from dataset name if not specified
    if dataset_type is None:
        if "humaneval" in dataset_name.lower() and "coop" not in dataset_name.lower():
            dataset_type = "humaneval"
        elif "coophumaneval" in dataset_name.lower() or "coop" in dataset_name.lower():
            dataset_type = "coophumaneval"
        else:
            raise ValueError(
                f"Could not infer dataset type from dataset name '{dataset_name}'. Please specify 'type' in dataset config."
            )
        print(f"Dataset type not specified, inferred as: {dataset_type}")

    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    # Get MAGRPO configuration (works for both single and multi-turn)
    magrpo_config = (
        config.get_section("magrpo") if hasattr(config, "get_section") else {}
    )

    # Check if this is multi-turn training
    num_turns = magrpo_config.get("num_turns", 1)
    is_multi_turn = num_turns > 1

    print(f"Multi-turn training enabled: num_turns={num_turns}") if is_multi_turn else print(
        f"Single-turn training: num_turns={num_turns}"
    )

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")

    # Use different output directory prefix for multi-turn
    if is_multi_turn:
        output_dir = os.path.join(output_base_dir, f"mt_job_{slurm_job_id}")
    else:
        output_dir = os.path.join(output_base_dir, f"job_{slurm_job_id}")

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)

    train_dataset = None
    eval_dataset = None
    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\nUsing model: {model_name}")
    print(f"Model type: {model_config.type}")
    print(f"Max context window: {model_config.max_length} tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = config.get("tokenizer.padding_side")
    if padding_side:
        tokenizer.padding_side = padding_side

    # Add special tokens if needed (e.g., FIM tokens for StarCoder)
    if model_config.special_tokens:
        print("Adding special tokens...")
        tokenizer.add_special_tokens(model_config.special_tokens)
        print(
            f"Special tokens added: {model_config.special_tokens.get('additional_special_tokens', [])}"
        )

    temperature = magrpo_config.get("temperature", model_config.temperature)
    top_p = magrpo_config.get("top_p", model_config.top_p)

    # External configuration (mode, sandbox, expert model, context flags)
    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}

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

    import re

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
        try:
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
        except Exception:
            pass

    if "train_dataset" in locals() and train_dataset is not None:
        _register_split(train_dataset)
    if "eval_dataset" in locals() and eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    # Use unified MAGRPOConfig which handles both single-turn and multi-turn
    magrpo_args = MAGRPOConfig(
        output_dir=output_dir,
        num_agents=magrpo_config.get("num_agents", 2),  # Pass num_agents to the config
        num_train_epochs=magrpo_config.get(
            "num_train_epochs", 10 if not is_multi_turn else 7
        ),
        per_device_train_batch_size=magrpo_config.get("per_device_train_batch_size", 1),
        learning_rate=magrpo_config.get("learning_rate", 1e-5),
        logging_steps=magrpo_config.get("logging_steps", 50),
        save_steps=magrpo_config.get("save_steps", 200),
        num_generations=magrpo_config.get("num_generations", 4),
        max_new_tokens=magrpo_config.get("max_new_tokens", 256),
        temperature=temperature,
        top_p=top_p,
        # Multi-turn parameters (automatically handled based on num_turns)
        num_turns=num_turns,
        discount=magrpo_config.get("discount", 0.9),
        joint_mode=magrpo_config.get("joint_mode", "cross"),
    )

    # Get appropriate formatters and functions based on dataset type, agent count, and training mode
    formatters = get_formatters(dataset_type, config.get("magrpo.num_agents", 2))
    reward_func = get_reward_function(dataset_type, config.get("magrpo.num_agents", 2))
    eval_logger, eval_aggregator = get_logger_and_aggregator(
        dataset_type, is_multi_turn
    )

    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    model_short_name = model_name.split("/")[-1].lower()

    # Use different wandb name for multi-turn
    if is_multi_turn:
        wandb_name = wandb_section.get("name", f"mt_magrpo_{dataset_type}")
    else:
        wandb_name = wandb_section.get("name", f"magrpo_{dataset_type}")

    # external_cfg already loaded above
    # Compute tags and add self-evolved when using analysis-based external modes
    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["magrpo", dataset_type or "code", f"turns_{num_turns}"]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    # Ensure list
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags
    if external_mode in ["level_passed", "level_feedback", "passed"]:
        if "self-evolved" not in tags:
            tags.append("self-evolved")

    # Collect full config sections for W&B searchability
    dataset_section = config.get_section("dataset") if hasattr(config, "get_section") else {}
    model_section = config.get_section("model") if hasattr(config, "get_section") else {}
    output_section = config.get_section("output") if hasattr(config, "get_section") else {}

    wandb_config = {
        "project": wandb_section.get("project", "mlrl"),
        "entity": wandb_section.get("entity", "nu-llpr"),
        "name": f"{wandb_name}_{model_short_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": tags,
        # Provide full sections for the trainer to log cleanly
        "config_sections": {
            "dataset": dataset_section,
            "model": model_section,
            "output": output_section,
            "external": external_cfg,
            "trainer": magrpo_config,
        },
    }

    # Get num_agents from magrpo config (where it belongs for MAGRPO training)
    num_agents = magrpo_config.get("num_agents", 2)
    agents = [
        AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_config.model_kwargs,
        )
        for _ in range(num_agents)
    ]

    reward_processor = None
    # Optional scale
    if config.get("reward_processor.enabled", False):
        scale_factor = config.get("reward_processor.scale_factor", 1)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
    # Optional shift via magrpo.reward_shift (default: -4 for code tasks)
    shift_val = magrpo_config.get("reward_shift", -4)
    if shift_val is not None:
        try:
            shift_val_f = float(shift_val)
        except (TypeError, ValueError):
            shift_val_f = None
        if shift_val_f is not None:
            shift_proc = RewardProcessors.shift(value=shift_val_f)
            if reward_processor is None:
                reward_processor = shift_proc
            else:
                # Compose scale then shift
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()

    trainer_kwargs = {
        "agents": agents,
        "num_agents": num_agents,
        "reward_func": reward_func,
        "formatters": formatters,
        "args": magrpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
        "dataset_type": dataset_type,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if (
        is_multi_turn
        and dataset_type
        and dataset_type.lower() in ["humaneval", "coophumaneval"]
    ):
        expert_model = external_cfg.get("expert_model", "deepseek-coder")
        # external_mode already loaded above

        def external_transition_wrapper(
            prompt, agent_completions, num_agents
        ):
            # Returns full next-turn prompts per agent (strings)
            # External prompt composition flags
            original_prompt_flag = external_cfg.get("original_prompt", True)
            previous_response_flag = external_cfg.get("previous_response", True)
            return get_external_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=num_agents,
                expert_model=expert_model,
                mode=external_mode,
                original_prompt=original_prompt_flag,
                previous_response=previous_response_flag,
            )

        trainer_kwargs["external_transition"] = external_transition_wrapper

    trainer = MAGRPOTrainer(**trainer_kwargs)
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
