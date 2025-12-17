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
from comlrl.trainers.maac import MAACConfig, MAACTrainer
from rewards.code_rewards import execution_reward_aux


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


def extract_problem_from_prompt(formatted_prompt: str) -> str:
    match = re.search(
        r"Problem:\s*(.*?)\n\nIMPORTANT INSTRUCTIONS:", formatted_prompt, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return formatted_prompt.strip()


def build_prompt_lookup(dataset) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for item in dataset:
        raw_prompt = item.get("prompt", "")
        if not raw_prompt:
            continue
        lookup[raw_prompt.strip()] = {
            "prompt": raw_prompt,
            "entry_point": item.get("entry_point", ""),
            "test": item.get("test", ""),
        }
    return lookup


def make_prompt_reward_fn(prompt_lookup: Dict[str, Dict[str, str]]):
    def _reward(
        prompts: List[str], aux_outputs: List[str], main_outputs: List[str]
    ) -> List[float]:
        if not prompts:
            return []

        problem_text = extract_problem_from_prompt(prompts[0])
        meta = prompt_lookup.get(problem_text) or prompt_lookup.get(problem_text.strip())
        if meta is None:
            raise KeyError("Failed to find metadata for provided prompt text.")

        count = min(len(aux_outputs), len(main_outputs))
        if count == 0:
            return []

        test_cases = [meta["test"]] * count
        entry_points = [meta["entry_point"]] * count
        raw_prompts = [meta["prompt"]] * count

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

    # ------------------------------------------------------------------ #
    # Config: load YAML and apply overrides
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Config: model, dataset, output
    # ------------------------------------------------------------------ #
    model_config = config.get_model_config()
    model_name = model_config.name
    dataset_name = config.get("dataset.name")
    dataset_type = config.get("dataset.type")
    dataset_split = config.get("dataset.split", "train")
    dataset_size = config.get("dataset.size")
    output_base_dir = config.get("output.base_dir", "output")
    output_verbose = config.get("output.verbose", False)

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

    # ------------------------------------------------------------------ #
    # MAAC-specific config (needed early for seed)
    # ------------------------------------------------------------------ #
    maac_cfg = config.get_section("maac") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", maac_cfg.get("seed", 42)))

    # ------------------------------------------------------------------ #
    # Output directory handling
    # ------------------------------------------------------------------ #
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"maac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(config, "save"):
        config.save(os.path.join(output_dir, "config.yaml"))

    # ------------------------------------------------------------------ #
    # Tokenizer / dataset
    # ------------------------------------------------------------------ #
    _set_seed(seed_value)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, split=dataset_split)
    if dataset_size is not None:
        usable = min(int(dataset_size), len(dataset))
        dataset = dataset.select(range(usable))
    else:
        usable = len(dataset)

    if output_verbose:
        print(f"Using model: {model_name}")
        print(f"Dataset: {dataset_name} split={dataset_split} size={usable}")

    formatters = build_prompt_formatters()
    prompt_lookup = build_prompt_lookup(dataset)
    reward_fn = make_prompt_reward_fn(prompt_lookup)

    # ------------------------------------------------------------------ #
    # MAAC-specific config
    # ------------------------------------------------------------------ #
    num_generations = maac_cfg.get("num_generations", maac_cfg.get("num_return_sequences", 1))
    use_sampling = maac_cfg.get("do_sample", num_generations > 1)
    top_k = maac_cfg.get("top_k")
    temperature = maac_cfg.get("temperature", model_config.temperature)
    top_p = maac_cfg.get("top_p", model_config.top_p)
    critic_model = (
        maac_cfg.get("critic_model")
        or maac_cfg.get("critic_model_name_or_path")
        or model_name
    )

    trainer = MAACTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=formatters,
        metrics_callback=None,
        args=MAACConfig(
            output_dir=os.path.join(output_dir, "maac"),
            actor_learning_rate=maac_cfg.get("actor_learning_rate", 5e-6),
            critic_learning_rate=maac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=maac_cfg.get("value_loss_coef", 0.6),
            rollout_buffer_size=maac_cfg.get("rollout_buffer_size", 8),
            mini_batch_size=maac_cfg.get("mini_batch_size", 4),
            ac_epochs=maac_cfg.get("ac_epochs", 1),
            max_new_tokens=maac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_train_epochs=maac_cfg.get("num_train_epochs", 20),
            per_device_train_batch_size=maac_cfg.get("per_device_train_batch_size", 1),
            num_agents=maac_cfg.get("num_agents", 2),
            num_return_sequences=num_generations,
            critic_model_name_or_path=critic_model,
        ),
        train_dataset=dataset,
        model_config={
            "tokenizer_kwargs": model_config.tokenizer_kwargs,
            "model_kwargs": model_config.model_kwargs,
            "critic_model_kwargs": maac_cfg.get(
                "critic_model_kwargs", model_config.model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(config, dataset_name, dataset_split, usable),
    )
    _set_seed(int(config.get("seed", maac_cfg.get("seed", 42))))
    trainer.train()

    if config.get("output.save_final_model", False):
        save_path = config.get("output.save_path", os.path.join(output_dir, "final_model"))
        trainer.save_model(save_path)
        if output_verbose:
            print(f"Model saved to: {save_path}")

    if wandb.run is not None:
        wandb.finish()


def _build_wandb_config(config: Config, dataset_name: str, dataset_split: str, size: int):
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    maac_section = config.get_section("maac") if hasattr(config, "get_section") else {}
    tags = wandb_section.get("tags", ["maac", dataset_name or "code", "turns_2"])
    return {
        "project": wandb_section.get("project", "maac"),
        "entity": wandb_section.get("entity"),
        "name": wandb_section.get("name", "maac_two_turn"),
        "dir": wandb_section.get("dir"),
        "tags": tags,
        "config_sections": {
            "dataset": {"name": dataset_name, "split": dataset_split, "size": size},
            "trainer": {
                "num_generations": maac_section.get("num_generations", 1),
                "max_new_tokens": maac_section.get("max_new_tokens", 256),
                "temperature": maac_section.get("temperature"),
                "top_p": maac_section.get("top_p"),
                "top_k": maac_section.get("top_k"),
            },
        },
    }


if __name__ == "__main__":
    main()
