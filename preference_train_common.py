import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import random
from typing import Any, Dict, Optional, Type

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from config import Config
from comlrl.utils.reward_processor import RewardProcessors
from loggers.ac_code_metrics import build_ac_code_metrics_callback
from train_magrpo import (
    get_formatters,
    get_logger_and_aggregator,
    get_reward_function,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_dataset_type(dataset_name: str, dataset_type: Optional[str]) -> str:
    if dataset_type is not None:
        return dataset_type
    lowered = dataset_name.lower()
    if "humaneval" in lowered and "coop" not in lowered:
        return "humaneval"
    if "coophumaneval" in lowered or "coop" in lowered:
        return "coophumaneval"
    raise ValueError(
        f"Could not infer dataset type from dataset name '{dataset_name}'. "
        "Please specify 'type' in dataset config."
    )


def build_reward_processor(config: Config):
    if not config.get("reward_processor.enabled", True):
        return None
    reward_processor = RewardProcessors.scale(
        factor=config.get("reward_processor.scale_factor", 1.0)
    )
    shift_val = config.get("reward_processor.shift", None)
    if shift_val is None:
        return reward_processor
    try:
        shift_value = float(shift_val)
    except (TypeError, ValueError):
        return reward_processor
    shift_proc = RewardProcessors.shift(value=shift_value)
    return (lambda p=reward_processor, s=shift_proc: (lambda x: s(p(x))))()


def run_preference_training(
    *,
    config: Config,
    section_name: str,
    args_cls: Type[Any],
    trainer_cls: Type[Any],
    algorithm_name: str,
) -> None:
    model_config = config.get_agent_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = infer_dataset_type(dataset_name, config.get("dataset.type"))
    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")
    output_base_dir = config.get("output.base_dir")
    output_verbose = bool(config.get("output.verbose", False))

    algo_config: Dict[str, Any] = config.get_section(section_name)
    seed_value = int(config.get("seed", algo_config.get("seed", 42)))
    set_seed(seed_value)

    num_agents = int(algo_config.get("num_agents", 2))
    agent_names = config.get("agents")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(
            isinstance(item, str) for item in agent_names
        ):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(item) for item in agent_names]

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config.save(os.path.join(output_dir, "config.yaml"))

    train_dataset = load_dataset(dataset_name, split=train_split)
    eval_dataset = load_dataset(dataset_name, split=eval_split)

    if output_verbose:
        display_model = (agent_names[0] if agent_names else model_name) or ""
        print(f"\nUsing model: {display_model}")
        print(f"Model type: {model_config.type}")
        print(f"Max context window: {model_config.max_length} tokens")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    if agent_names:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in agent_names]
    else:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer_source)]

    for tokenizer in tokenizers:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        padding_side = config.get("tokenizer.padding_side")
        if padding_side:
            tokenizer.padding_side = padding_side
        if model_config.special_tokens:
            tokenizer.add_special_tokens(model_config.special_tokens)

    args_kwargs = dict(algo_config)
    args_kwargs.setdefault("temperature", model_config.temperature)
    args_kwargs.setdefault("top_p", model_config.top_p)
    args_kwargs.setdefault("top_k", model_config.top_k)
    args_kwargs.setdefault("num_agents", num_agents)
    args = args_cls(**args_kwargs)

    formatters = get_formatters(dataset_type, num_agents)
    reward_func = get_reward_function(dataset_type, num_agents)
    eval_logger, eval_aggregator = get_logger_and_aggregator(
        dataset_type,
        bool(getattr(args, "num_turns", 1) > 1),
    )
    metrics_callback = None
    if algorithm_name.lower() == "marlhf":
        metrics_callback = build_ac_code_metrics_callback(
            num_agents,
            int(getattr(args, "num_turns", 1)),
        )

    wandb_section = config.get_section("wandb")
    default_name = f"{dataset_type}-{algorithm_name.lower()}"
    wandb_name = (
        wandb_section.get("name") or wandb_section.get("run_name") or default_name
    )
    default_tags = [
        algorithm_name.lower(),
        dataset_type or "code",
        "multi-agent",
        f"turns_{getattr(args, 'num_turns', 1)}",
    ]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags

    wandb_config = {
        "project": wandb_section.get("project", "comlrl"),
        "entity": wandb_section.get("entity", "OpenMLRL"),
        "name": f"{wandb_name}",
        "dir": wandb_section.get("dir", output_base_dir),
        "tags": tags,
        "config_sections": {
            "dataset": config.get_section("dataset"),
            "agent_model": config.get_section("agent_model"),
            "output": config.get_section("output"),
            "external": config.get_section("external"),
            "trainer": algo_config,
        },
    }

    import external as external_mod
    import rewards.code_rewards as code_rewards

    code_rewards.VERBOSE = bool(output_verbose)
    external_mod.VERBOSE = bool(output_verbose)

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
        "formatters": formatters,
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
        "dataset_type": dataset_type,
        "args": args,
    }
    if metrics_callback is not None:
        trainer_kwargs["metrics_callback"] = metrics_callback

    reward_processor = build_reward_processor(config)
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    trainer = trainer_cls(**trainer_kwargs)
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get(
            "output.save_path",
            os.path.join(output_dir, "final_model"),
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")
