"""
Train MADPO with the same data, reward, formatter, and external-transition setup
used by train_magrpo.py.
"""

import argparse
import os
import random
import re
import sys
from dataclasses import fields
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.reinforce import MADPOConfig, MADPOTrainer
from comlrl.utils.reward_processor import RewardProcessors
from config import Config, add_config_args, parse_overrides
import external as external_ctx
from external import get_external_transition
from rewards.code_rewards import make_code_oracle_reward_function
from train_magrpo import get_formatters, get_logger_and_aggregator


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_dataset_type(dataset_name: str) -> str:
    name = dataset_name.lower()
    if "humaneval" in name and "coop" not in name:
        return "humaneval"
    if "coophumaneval" in name or "coop" in name:
        return "coophumaneval"
    if "mbpp" in name:
        return "mbpp"
    raise ValueError(
        f"Could not infer dataset type from dataset name '{dataset_name}'."
    )


def _normalize_prompt(prompt: str) -> str:
    return " ".join((prompt or "").split()).strip()


def _parse_sandbox_slice(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "all":
            return 0
        if lowered.lstrip("-").isdigit():
            return int(lowered)
        return None
    if isinstance(value, int):
        return value
    return None if value is None else 0


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

    search_start = check_idx + 1 if check_idx is not None else 0
    asserts = []
    for line in lines[search_start:]:
        stripped = line.strip()
        if stripped.startswith("assert") and "candidate" in stripped:
            asserts.append(stripped)
    if not asserts:
        return test_code

    new_parts = []
    preamble_text = "\n".join(preamble).strip()
    if preamble_text:
        new_parts.append(preamble_text)
    new_parts.append("def check(candidate):")
    selected = asserts[:n] if n > 0 else asserts[n:]
    for assertion in selected:
        new_parts.append(f"    {assertion}")
    return "\n".join(new_parts) + "\n"


def _register_external_context(train_dataset, eval_dataset, external_cfg: Dict[str, Any]):
    sandbox_slice = _parse_sandbox_slice(external_cfg.get("sandbox_slice", 1))
    context_map = {}

    def register_split(dataset):
        for item in dataset:
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
        register_split(train_dataset)
    if eval_dataset is not None:
        register_split(eval_dataset)

    external_ctx.set_context_resolver(
        lambda prompt: context_map.get(_normalize_prompt(prompt))
    )


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _build_reward_processor(config: Config):
    if not config.get("reward_processor.enabled", True):
        return None
    reward_processor = RewardProcessors.scale(
        factor=config.get("reward_processor.scale_factor", 1.0)
    )
    shift_val = config.get("reward_processor.shift", None)
    if shift_val is None:
        return reward_processor
    try:
        shift_val_f = float(shift_val)
    except (TypeError, ValueError):
        return reward_processor
    shift_proc = RewardProcessors.shift(value=shift_val_f)
    prev = reward_processor
    return (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MADPO.")
    add_config_args(parser)
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Please provide --config.")

    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    model_config = config.get_agent_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type") or _infer_dataset_type(dataset_name)
    output_base_dir = config.get("output.base_dir", "output_madpo")
    output_verbose = bool(config.get("output.verbose", False))

    magrpo_section = (
        config.get_section("magrpo") if hasattr(config, "get_section") else {}
    )
    madpo_section = (
        config.get_section("madpo") if hasattr(config, "get_section") else {}
    )
    trainer_section = dict(magrpo_section)
    trainer_section.update(madpo_section)
    preference_source = str(trainer_section.get("preference_source", "online")).lower()
    if preference_source == "offline" and not trainer_section.get(
        "preference_buffer_path"
    ):
        buffer_path = config.get("preference.output_path")
        if buffer_path:
            trainer_section["preference_buffer_path"] = buffer_path

    seed_value = int(config.get("seed", trainer_section.get("seed", 42)))
    num_turns = int(trainer_section.get("num_turns", 2))
    num_agents = int(trainer_section.get("num_agents", 2))
    is_multi_turn = num_turns > 1

    train_dataset = load_dataset(dataset_name, split=config.get("dataset.train_split"))
    eval_dataset = load_dataset(dataset_name, split=config.get("dataset.eval_split"))

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(
        output_base_dir, f"{'mt_' if is_multi_turn else ''}madpo_job_{slurm_job_id}"
    )
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(config, "save"):
        config.save(os.path.join(output_dir, "config.yaml"))

    _set_seed(seed_value)

    agent_names = config.get("agents")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(
            isinstance(name, str) for name in agent_names
        ):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(name) for name in agent_names]

    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    tokenizers = (
        [AutoTokenizer.from_pretrained(name) for name in agent_names]
        if agent_names
        else [AutoTokenizer.from_pretrained(tokenizer_source)]
    )
    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tokenizer.padding_side = padding_side
        if model_config.special_tokens:
            tokenizer.add_special_tokens(model_config.special_tokens)

    external_cfg = (
        config.get_section("external") if hasattr(config, "get_section") else {}
    )
    external_prompt_passthrough = _bool_value(
        external_cfg.get("external_prompt_passthrough", False)
    )
    _register_external_context(train_dataset, eval_dataset, external_cfg)

    allowed_fields = {field.name for field in fields(MADPOConfig)}
    madpo_kwargs = {
        key: value for key, value in trainer_section.items() if key in allowed_fields
    }
    madpo_kwargs.update(
        {
            "num_agents": num_agents,
            "num_turns": num_turns,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "external_prompt_passthrough": external_prompt_passthrough,
        }
    )
    madpo_args = MADPOConfig(**madpo_kwargs)

    formatters = get_formatters(dataset_type, num_agents)
    reward_func = make_code_oracle_reward_function(num_agents)
    eval_logger, eval_aggregator = get_logger_and_aggregator(
        dataset_type, is_multi_turn
    )

    import rewards.code_rewards as code_rewards
    import external as external_mod

    code_rewards.VERBOSE = bool(output_verbose)
    external_mod.VERBOSE = bool(output_verbose)

    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["madpo", dataset_type or "code", f"turns_{num_turns}"]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags
    if "madpo" not in tags:
        tags.insert(0, "madpo")
    source_tag = f"pref_{preference_source}"
    if source_tag not in tags:
        tags.append(source_tag)
    if external_mode == "level_feedback" and "self-evolved" not in tags:
        tags.append("self-evolved")

    dataset_section = (
        config.get_section("dataset") if hasattr(config, "get_section") else {}
    )
    model_section = (
        config.get_section("agent_model") if hasattr(config, "get_section") else {}
    )
    output_section = (
        config.get_section("output") if hasattr(config, "get_section") else {}
    )
    wandb_name = (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or f"{dataset_type}-madpo"
    )
    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": wandb_name,
        "dir": wandb_section.get("dir", output_base_dir),
        "tags": tags,
        "config_sections": {
            "dataset": dataset_section,
            "agent_model": model_section,
            "output": output_section,
            "external": external_cfg,
            "trainer": trainer_section,
        },
    }

    trainer_kwargs = {
        "agent_model": model_name or None,
        "agents": agent_names,
        "num_agents": num_agents,
        "tokenizer": tokenizers if agent_names else tokenizers[0],
        "model_config": {
            "torch_dtype": model_config.torch_dtype,
            "special_tokens": model_config.special_tokens,
        },
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "reward_func": reward_func,
        "reward_processor": _build_reward_processor(config),
        "formatters": formatters,
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
        "dataset_type": dataset_type,
        "args": madpo_args,
    }

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

    trainer = MADPOTrainer(**trainer_kwargs)
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
