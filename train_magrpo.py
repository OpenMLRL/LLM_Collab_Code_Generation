"""
Unified training script for MAGRPO that supports both single-turn and multi-turn training.
The training mode is determined by the num_turns parameter in the config file.
Supports multiple datasets and configurations via YAML files.
"""

import argparse
import os
import re
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import loggers for different datasets
from loggers.code_logger import (
    aggregate_code_metrics_for_logging,
    code_reward_logger,
)
from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

from rewards.code_rewards import execution_reward_humaneval_aux
from comlrl.utils.reward_processor import RewardProcessors
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from external import get_expert_feedback


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
    Formatter for the auxiliary function generator (Agent 1) for code tasks.
    Optionally includes expert feedback for multi-turn training.
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
    Formatter for the main function generator (Agent 2) for code tasks.
    Optionally includes expert feedback for multi-turn training.
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




def get_formatters(dataset_type: str):
    """Get the appropriate formatters based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
        )

    formatters_map = {
        "humaneval": [aux_function_formatter, main_function_formatter],
        "coophumaneval": [aux_function_formatter, main_function_formatter],
    }
    return formatters_map.get(
        dataset_type.lower(), [aux_function_formatter, main_function_formatter]
    )


def get_logger_and_aggregator(dataset_type: str, is_multi_turn: bool = False):
    """
    Get the appropriate logger and aggregator functions based on dataset type.
    For multi-turn training with code datasets, use the multi-turn logger.
    """
    if dataset_type is None:
        return None, None

    # For multi-turn training with code datasets, use multi-turn logger
    if is_multi_turn and dataset_type.lower() in ["humaneval", "coophumaneval"]:
        return mt_humaneval_logger, aggregate_mt_humaneval_metrics_for_logging

    # Standard single-turn loggers
    logger_map = {
        "humaneval": (code_reward_logger, aggregate_code_metrics_for_logging),
        "coophumaneval": (code_reward_logger, aggregate_code_metrics_for_logging),
    }

    return logger_map.get(dataset_type.lower(), (None, None))


def get_reward_function(dataset_type: str):
    """Get the appropriate reward function based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
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
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function to run the unified MAGRPO training."""
    parser = argparse.ArgumentParser(
        description="Train MAGRPO with configurable dataset (single-turn or multi-turn)"
    )
    add_config_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name to use (overrides config)",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory (overrides config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        default=None,
        help="Number of turns for multi-turn training (overrides config)",
    )
    parser.add_argument(
        "--turn_gradient_weights",
        type=float,
        nargs="+",
        default=None,
        help="Turn gradient weights for multi-turn training (overrides config)",
    )

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    # Apply command-line overrides
    if args.model_name:
        config.update({"model_name": args.model_name})
    if args.output_base_dir:
        config.update({"output": {"base_dir": args.output_base_dir}})
    if args.num_epochs is not None:
        config.update({"magrpo": {"num_train_epochs": args.num_epochs}})
    if args.num_turns is not None:
        config.update({"magrpo": {"num_turns": args.num_turns}})
    if args.turn_gradient_weights is not None:
        config.update({"magrpo": {"turn_gradient_weights": args.turn_gradient_weights}})

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
    
    # Validate turn gradient weights for multi-turn
    if is_multi_turn:
        turn_weights = magrpo_config.get("turn_gradient_weights", [1.0] * num_turns)
        if len(turn_weights) != num_turns:
            raise ValueError(
                f"turn_gradient_weights must have {num_turns} values, got {len(turn_weights)}"
            )
        print(f"Multi-turn training enabled: num_turns={num_turns}, weights={turn_weights}")
    else:
        print(f"Single-turn training: num_turns={num_turns}")

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
    
    # Use unified MAGRPOConfig which handles both single-turn and multi-turn
    magrpo_args = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=magrpo_config.get("num_train_epochs", 10 if not is_multi_turn else 7),
        per_device_train_batch_size=magrpo_config.get("per_device_train_batch_size", 1),
        learning_rate=magrpo_config.get("learning_rate", 1e-5),
        logging_steps=magrpo_config.get("logging_steps", 50),
        save_steps=magrpo_config.get("save_steps", 200),
        num_generations=magrpo_config.get("num_generations", 4),
        max_new_tokens=magrpo_config.get("max_new_tokens", 256),
        temperature=temperature,
        top_p=top_p,
        beta=magrpo_config.get("beta", 0.02),
        # Multi-turn parameters (automatically handled based on num_turns)
        num_turns=num_turns,
        turn_gradient_weights=magrpo_config.get("turn_gradient_weights", [1.0] * num_turns),
        early_termination_weight=magrpo_config.get("early_termination_weight", 2.0),
        expert_model=magrpo_config.get("expert_model", "claude-3-5-sonnet-20241022"),
    )

    # Get appropriate formatters and functions based on dataset type and training mode
    formatters = get_formatters(dataset_type)
    reward_func = get_reward_function(dataset_type)
    eval_logger, eval_aggregator = get_logger_and_aggregator(dataset_type, is_multi_turn)

    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    model_short_name = model_name.split("/")[-1].lower()
    
    # Use different wandb name for multi-turn
    if is_multi_turn:
        wandb_name = wandb_section.get("name", f"mt_magrpo_{dataset_type}")
    else:
        wandb_name = wandb_section.get("name", f"magrpo_{dataset_type}")

    wandb_config = {
        "project": wandb_section.get("project", "mlrl"),
        "entity": wandb_section.get("entity", "nu-llpr"),
        "name": f"{wandb_name}_{model_short_name}",
        "dir": wandb_section.get("dir", "../../../projects/bepg/sliu30"),
        "tags": wandb_section.get("tags", ["magrpo", dataset_type or "code", f"turns_{num_turns}"]),
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
        "args": magrpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "eval_logger": eval_logger,
        "eval_aggregator": eval_aggregator,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processors"] = reward_processor
    
    # Add external_transition for code tasks if multi-turn is enabled
    if is_multi_turn and dataset_type and dataset_type.lower() in ["humaneval", "coophumaneval"]:
        # Create a wrapper that provides test and expert_model from batch_item and config
        expert_model = magrpo_config.get("expert_model", "claude-3-5-sonnet-20241022")
        
        def external_transition_wrapper(prompt, best_reward, aux_completion, main_completion, batch_item):
            """Wrapper that adds test and expert_model from batch_item and config."""
            return get_expert_feedback(
                prompt=prompt,
                test=batch_item.get("test", ""),
                best_reward=best_reward,
                aux_completion=aux_completion,
                main_completion=main_completion,
                expert_model=expert_model,
            )
        
        trainer_kwargs["external_transition"] = external_transition_wrapper

    # Use the unified MAGRPOTrainer which automatically handles single/multi-turn based on config
    trainer = MAGRPOTrainer(**trainer_kwargs)

    trainer.train()

    save_final = config.get("output.save_final_model", True)
    if save_final:
        save_path = config.get(
            "output.save_path", os.path.join(output_dir, "final_model")
        )
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()