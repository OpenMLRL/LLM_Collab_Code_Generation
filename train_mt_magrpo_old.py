import argparse
import ast
import os
import re
import sys

import numpy as np
import torch
import wandb

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)
from rewards.arxiv_rewards import arxiv_combined_reward
from rewards.code_rewards import execution_reward_humaneval_aux
from rewards.tldr_rewards import tldr_combined_reward
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer


def extract_function_params_from_prompt(prompt_text):
    """Extract function parameters from the prompt text."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def aux_function_formatter(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """
    Formatter for the auxiliary function generator (Agent 1).
    Optionally includes expert feedback.
    """
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

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

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def main_function_formatter(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """
    Formatter for the main function generator (Agent 2).
    Optionally includes expert feedback.
    """
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

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def background_agent_formatter_mt(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """Formatter for the background agent (Agent 1) for ArXiv dataset."""
    abstract = example.get("abstract_text", "")

    if not abstract:
        return "Error: No abstract provided."

    prompt_text = f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide methodology and implications
- You just need to focus on background and motivation
- Avoid repeating methodology and implications content
"""

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def complementary_agent_formatter_mt(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """Formatter for the complementary agent (Agent 2) for ArXiv dataset."""
    abstract = example.get("abstract_text", "")

    if not abstract:
        return "Error: No abstract provided."

    prompt_text = f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide the background and motivation
- You just need to focus on methodology and implications
- Avoid repeating background and motivation content
"""

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def summary_agent_formatter_mt(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """Formatter for the summary agent (Agent 1) for TLDR dataset."""
    prompt = example.get("prompt", "")

    if not prompt:
        return "Error: No prompt provided."

    prompt_text = f"""Create a concise summary response to this post.

Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Provide a brief, focused summary in one sentence or a few sentences
- Be factual and informative
"""

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def elaboration_agent_formatter_mt(
    example: Dict[str, Any], expert_feedback: Optional[str] = None
) -> str:
    """Formatter for the elaboration agent (Agent 2) for TLDR dataset."""
    prompt = example.get("prompt", "")

    if not prompt:
        return "Error: No prompt provided."

    prompt_text = f"""Create a detailed summary response to this post.

Original Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Use more unique words
- Use some transition words to improve flow
"""

    if expert_feedback is not None:
        prompt_text += f"\n\nHere is the feedback from an expert:\n{expert_feedback}"

    return prompt_text


def get_formatters(dataset_type: str):
    """Get the appropriate formatters based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval/arxiv/tldr' to the dataset section."
        )

    formatters_map = {
        "humaneval": [aux_function_formatter, main_function_formatter],
        "coophumaneval": [aux_function_formatter, main_function_formatter],
        "arxiv": [background_agent_formatter_mt, complementary_agent_formatter_mt],
        "tldr": [summary_agent_formatter_mt, elaboration_agent_formatter_mt],
    }
    return formatters_map.get(
        dataset_type.lower(), [aux_function_formatter, main_function_formatter]
    )


def get_reward_function(dataset_type: str, train_dataset):
    """Get the appropriate reward function based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval/arxiv/tldr' to the dataset section."
        )

    if dataset_type.lower() in ["humaneval", "coophumaneval"]:

        def reward_wrapper(completion1, completion2, batch_items=None, prompts=None):
            batch_size = len(completion1)

            test_cases = []
            entry_points = []
            original_prompts = []

            if batch_items is not None:
                for item in batch_items:
                    test_cases.append(item["test"])
                    entry_points.append(item["entry_point"])
                    original_prompts.append(item["prompt"])
                    print(f"Using passed batch item: {item['entry_point']}")
            else:
                raise ValueError("batch_items must be provided for reward calculation")

            return execution_reward_humaneval_aux(
                completion1, completion2, test_cases, entry_points, original_prompts
            )

        return reward_wrapper
    elif dataset_type.lower() == "arxiv":
        return arxiv_combined_reward
    elif dataset_type.lower() == "tldr":
        return tldr_combined_reward
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MT-MAGRPO model with customizable parameters"
    )
    add_config_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use (overrides config)",
    )

    parser.add_argument(
        "--num_epochs",
        "--num_epoch",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--turn_gradient_weights",
        type=float,
        nargs="+",
        default=None,
        help="Turn gradient weights as a list of floats (overrides config)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size per device (overrides config)",
    )

    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Number of generations for MAGRPO (overrides config)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation (overrides config)",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta parameter for MAGRPO (overrides config)",
    )

    return parser.parse_args()


def main():
    """Main function to run the multi-turn experiment with expert feedback."""
    args = parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = (
            Path(__file__).parent / "configs" / "mt_magrpo_he_config.yaml"
        )
        if default_config_path.exists():
            config = ConfigLoader(str(default_config_path))
        else:
            from types import SimpleNamespace

            config = SimpleNamespace()
            config.get = lambda key, default=None: {
                "model_name": "Qwen/Qwen2.5-Coder-3B",
                "output.base_dir": "../../../projects/bepg/sliu30/output_mt",
                "dataset.name": "openai/openai_humaneval",
                "dataset.train_split": "test[:50]",
                "dataset.eval_split": "test[50:66]",
            }.get(key, default)
            config.get_section = lambda section: {}
            config.save = lambda path: None

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    if args.model_name:
        config.update({"model_name": args.model_name})
    if args.num_epochs is not None:
        config.update({"mt_magrpo": {"num_train_epochs": args.num_epochs}})
    if args.batch_size is not None:
        config.update({"mt_magrpo": {"per_device_train_batch_size": args.batch_size}})
    if args.learning_rate is not None:
        config.update({"mt_magrpo": {"learning_rate": args.learning_rate}})
    if args.beta is not None:
        config.update({"mt_magrpo": {"beta": args.beta}})
    if args.num_generations is not None:
        config.update({"mt_magrpo": {"num_generations": args.num_generations}})
    if args.temperature is not None:
        config.update({"mt_magrpo": {"temperature": args.temperature}})
    if args.turn_gradient_weights is not None:
        config.update(
            {"mt_magrpo": {"turn_gradient_weights": args.turn_gradient_weights}}
        )

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
        elif "arxiv" in dataset_name.lower():
            dataset_type = "arxiv"
        elif "tldr" in dataset_name.lower():
            dataset_type = "tldr"
        else:
            # Default to humaneval for backward compatibility
            dataset_type = "humaneval"
            print(f"Dataset type not specified, defaulting to: {dataset_type}")
        if dataset_type != "humaneval":
            print(f"Dataset type not specified, inferred as: {dataset_type}")
    train_split = config.get("dataset.train_split")
    eval_split = config.get("dataset.eval_split")

    mt_magrpo_config = (
        config.get_section("mt_magrpo") if hasattr(config, "get_section") else {}
    )
    turn_weights = mt_magrpo_config.get("turn_gradient_weights", [1.2, 0.8])
    if len(turn_weights) != 2:
        raise ValueError(
            f"turn_gradient_weights must have exactly 2 values, got {len(turn_weights)}"
        )

    print(f"Running with num_epochs: {mt_magrpo_config.get('num_train_epochs', 7)}")
    print(f"Running with turn_gradient_weights: {turn_weights}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")

    output_dir = os.path.join(output_base_dir, f"mt_expert_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(config, "save"):
        config_save_path = os.path.join(output_dir, "config.yaml")
        config.save(config_save_path)
        print(f"Configuration saved to: {config_save_path}")

    try:
        train_dataset = load_dataset(dataset_name, split=train_split)
        eval_dataset = load_dataset(dataset_name, split=eval_split)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")

    print(f"\nUsing model: {model_name}")
    print(f"Model type: {model_config.type}")
    print(f"Max context window: {model_config.max_length} tokens")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens if needed (e.g., FIM tokens for StarCoder)
    if model_config.special_tokens:
        print("Adding special tokens...")
        tokenizer.add_special_tokens(model_config.special_tokens)
        print(
            f"Special tokens added: {model_config.special_tokens.get('additional_special_tokens', [])}"
        )

    temperature = mt_magrpo_config.get("temperature", model_config.temperature)
    top_p = mt_magrpo_config.get("top_p", model_config.top_p)

    mt_config = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=mt_magrpo_config.get("num_train_epochs", 7),
        per_device_train_batch_size=mt_magrpo_config.get(
            "per_device_train_batch_size", 1
        ),
        learning_rate=mt_magrpo_config.get("learning_rate", 1e-5),
        logging_steps=mt_magrpo_config.get("logging_steps", 50),
        save_steps=mt_magrpo_config.get("save_steps", 200),
        num_generations=mt_magrpo_config.get("num_generations", 4),
        max_new_tokens=mt_magrpo_config.get("max_new_tokens", 256),
        temperature=temperature,
        top_p=top_p,
        beta=mt_magrpo_config.get("beta", 0.02),
        num_turns=mt_magrpo_config.get("num_turns", 2),
        turn_gradient_weights=turn_weights,
        early_termination_weight=mt_magrpo_config.get("early_termination_weight", 2.0),
        expert_model=mt_magrpo_config.get("expert_model", "claude-3-5-sonnet-20241022"),
    )

    formatters = get_formatters(dataset_type)
    reward_func = get_reward_function(dataset_type, train_dataset)

    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    model_short_name = model_name.split("/")[-1].lower()
    wandb_name = wandb_section.get("name", f"mt_magrpo_{dataset_type}")

    wandb_config = {
        "project": wandb_section.get("project", "mlrl"),
        "entity": wandb_section.get("entity", "nu-llpr"),
        "name": f"{wandb_name}_{model_short_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": wandb_section.get(
            "tags", ["mt_magrpo", dataset_type, "multi-agent", "multi-turn"]
        ),
    }

    agents_config = (
        config.get_section("agents") if hasattr(config, "get_section") else {}
    )
    num_agents = agents_config.get("num_agents", 2)

    print(f"\nCreating {num_agents} agents with {model_name}...")

    agents = [
        AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_config.model_kwargs,
        )
        for _ in range(num_agents)
    ]
    print("Agents created successfully!")

    reward_processor = None
    if config.get("reward_processor.enabled", False):
        scale_factor = config.get("reward_processor.scale_factor", 1)
        reward_processor = RewardProcessors.scale(factor=scale_factor)

    trainer_kwargs = {
        "agents": agents,
        "num_agents": num_agents,
        "reward_funcs": reward_func,
        "formatters": formatters,
        "args": mt_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "eval_logger": mt_humaneval_logger,
        "eval_aggregator": aggregate_mt_humaneval_metrics_for_logging,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processors"] = reward_processor

    trainer = MAGRPOTrainer(**trainer_kwargs)

    trainer.train()

    save_final = config.get("output.save_final_model", True)
    if save_final:
        save_path = os.path.join(output_dir, "final_model")
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
