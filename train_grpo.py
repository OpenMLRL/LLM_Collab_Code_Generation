"""
Generic single-agent training script for GRPO that supports multiple datasets and configurations.
Now also supports multi-turn training with external transitions (similar to MAGRPO),
by adapting external prompts for a single agent across turns.
Uses YAML configuration files to define all parameters.
"""

import argparse
import os
import re
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from typing import Any, Dict

from config import Config, add_config_args, parse_overrides
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def complete_function_formatter(example: Dict[str, Any]) -> str:
    """
    Formatter for the complete function generator (single agent).
    Explicitly instructs to output ONLY the function code.
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


def execution_reward_single_agent(completions, batch_items=None):
    """
    Compute execution-based rewards for single agent completions.
    Adapts to use execution_reward_aux by treating single agent
    completion as the main function with no aux function.
    """
    # Adapt single-agent completions to multi-agent reward function format
    completion1_list = []
    completion2_list = []
    test_cases = []
    entry_points = []
    prompts = []

    for idx, completion in enumerate(completions):
        if batch_items and idx < len(batch_items):
            completion1_list.append("")  # No aux function for single agent
            completion2_list.append(completion)

            test_cases.append(batch_items[idx]["test"])
            entry_points.append(batch_items[idx]["entry_point"])
            prompts.append(batch_items[idx].get("prompt", ""))
        else:
            completion1_list.append("")
            completion2_list.append("")
            test_cases.append("")
            entry_points.append("")
            prompts.append("")

    raw_rewards = execution_reward_aux(
        completion1_list, completion2_list, test_cases, entry_points, prompts
    )
    return raw_rewards


## Removed dead factory create_execution_reward_function (unused)


def get_formatter(dataset_type: str):
    """Get the appropriate formatter based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
        )

    formatters_map = {
        "humaneval": complete_function_formatter,
        "coophumaneval": complete_function_formatter,
    }
    return formatters_map.get(dataset_type.lower(), complete_function_formatter)


def get_reward_function(dataset_type: str):
    """Get the appropriate reward function based on dataset type."""
    if dataset_type is None:
        raise ValueError(
            "dataset.type not specified in config. Please add 'type: humaneval/coophumaneval' to the dataset section."
        )

    if dataset_type.lower() in ["humaneval", "coophumaneval"]:
        return execution_reward_single_agent
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main function to run the single agent experiment."""
    parser = argparse.ArgumentParser(description="Train GRPO with configurable dataset")
    add_config_args(parser)

    

    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "grpo_he_config.yaml"
        if default_config_path.exists():
            config = Config(str(default_config_path))
        else:
            raise ValueError("Please provide a configuration file using --config")

    if args.override:
        overrides = parse_overrides(args.override)
        config.update(overrides)

    

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

    # Read GRPO section early (for multi-turn flags)
    grpo_config = config.get_section("grpo") if hasattr(config, "get_section") else {}

    # Determine single vs multi-turn
    num_turns = grpo_config.get("num_turns", 1)
    is_multi_turn = num_turns > 1

    output_verbose = config.get("output.verbose", True)
    if output_verbose:
        print(f"Multi-turn GRPO enabled: num_turns={num_turns}") if is_multi_turn else print(
            f"Single-turn GRPO: num_turns={num_turns}"
        )

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    # Use different output directory prefix for multi-turn for clarity
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
        return

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

    print(f"\nLoading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_config.model_kwargs,
    )
    print("Model loaded successfully!")

    temperature = grpo_config.get("temperature", model_config.temperature)
    top_p = grpo_config.get("top_p", model_config.top_p)

    # External configuration (mode, sandbox, expert model, context flags)
    external_cfg = config.get_section("external") if hasattr(config, "get_section") else {}

    # Register external context resolver using dataset items (for external modes)
    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    context_map: Dict[str, Any] = {}

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

    import re as _re

    def _make_sliced_assert_tests(test_code: str, n: int) -> str:
        if not isinstance(test_code, str) or not test_code.strip():
            return test_code
        if n is None or n == 0:
            return test_code
        lines = test_code.splitlines()
        preamble = []
        check_idx = None
        for idx, line in enumerate(lines):
            if _re.match(r"\s*def\s+check\s*\(candidate\)\s*:\s*", line):
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

    grpo_args = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=grpo_config.get("num_train_epochs", 10),
        per_device_train_batch_size=grpo_config.get("per_device_train_batch_size", 1),
        learning_rate=grpo_config.get("learning_rate", 1e-5),
        logging_steps=grpo_config.get("logging_steps", 50),
        save_steps=grpo_config.get("save_steps", 200),
        num_generations=grpo_config.get("num_generations", 4),
        max_new_tokens=grpo_config.get("max_new_tokens", 256),
        temperature=temperature,
        top_p=top_p,
        # Multi-turn parameters
        num_turns=num_turns,
        discount=grpo_config.get("discount", 0.9),
        joint_mode=grpo_config.get("joint_mode", "aligned"),
        termination_threshold=grpo_config.get("termination_threshold", None),
        # GRPO-style advantage params
        normalize_advantage=grpo_config.get("normalize_advantage", False),
        epsilon_clip=grpo_config.get("epsilon_clip", None),
    )

    formatter = get_formatter(dataset_type)
    reward_func = get_reward_function(dataset_type)

    wandb_section = (
        config.get_section("wandb") if hasattr(config, "get_section") else {}
    )
    model_short_name = model_name.split("/")[-1].lower()
    # Use different wandb name for multi-turn
    if is_multi_turn:
        wandb_name = wandb_section.get("name", f"mt_grpo_{dataset_type}")
    else:
        wandb_name = wandb_section.get("name", f"grpo_{dataset_type}")

    # external_cfg already loaded above
    # Compute tags and add self-evolved when using analysis-based external modes
    external_mode = external_cfg.get("mode", "level_feedback")
    default_tags = ["grpo", dataset_type or "code", f"turns_{num_turns}"]
    tags_from_cfg = wandb_section.get("tags", default_tags)
    tags = list(tags_from_cfg) if isinstance(tags_from_cfg, list) else default_tags
    if external_mode in ["level_passed", "level_feedback", "passed"]:
        if "self-evolved" not in tags:
            tags.append("self-evolved")

    # If sandbox_slice is active (non-zero), append _slice to run name
    if isinstance(sandbox_slice, int) and sandbox_slice != 0:
        if not str(wandb_name).endswith("_slice"):
            wandb_name = f"{wandb_name}_slice"
        if "slice" not in tags:
            tags.append("slice")

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
            "trainer": grpo_config,
        },
    }

    # Propagate verbosity to reward/external modules
    try:
        import rewards.code_rewards as code_rewards
        code_rewards.VERBOSE = bool(output_verbose)
    except Exception:
        pass
    try:
        import external as external_mod
        external_mod.VERBOSE = bool(output_verbose)
    except Exception:
        pass

    reward_processor = None
    # Optional scale
    if config.get("reward_processor.enabled", False):
        scale_factor = config.get("reward_processor.scale_factor", 1)
        reward_processor = RewardProcessors.scale(factor=scale_factor)
    # Optional shift via grpo.reward_shift (default: -2.1 for single-agent code tasks)
    shift_val = grpo_config.get("reward_shift", -2.1)
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
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()

    # Use agents=[model] to keep dtype and loading behavior aligned with MAGRPO
    trainer_kwargs = {
        "agents": [model],
        "num_agents": 1,
        "reward_func": reward_func,
        "formatters": formatter,
        "args": grpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "dataset_type": dataset_type,
    }

    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    # Multi-turn external transition support for single-agent GRPO
    if (
        is_multi_turn
        and dataset_type
        and dataset_type.lower() in ["humaneval", "coophumaneval"]
    ):
        expert_model = external_cfg.get("expert_model", "deepseek-coder")

        def external_transition_wrapper(
            prompt, agent_completions, num_agents
        ):
            # Single-agent: pass prior main completion; aux is empty internally
            main_best = agent_completions[0] if agent_completions else ""

            original_prompt_flag = external_cfg.get("original_prompt", True)
            previous_response_flag = external_cfg.get("previous_response", True)
            prompts = get_external_transition(
                prompt=prompt,
                agent_completions=[main_best],
                num_agents=1,
                expert_model=expert_model,
                mode=external_mode,
                original_prompt=original_prompt_flag,
                previous_response=previous_response_flag,
            )

            # Ensure list of one string is returned
            if isinstance(prompts, (list, tuple)):
                return list(prompts)
            return [str(prompts)]

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
