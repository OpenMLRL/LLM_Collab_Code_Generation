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
from comlrl.trainers.iac import IACConfig, IACTrainer
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
        description="Independent Actor-Critic training for cooperative code generation."
    )
    add_config_args(parser)
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Config: load YAML and apply overrides
    # ------------------------------------------------------------------ #
    if args.config:
        config = Config(args.config)
    else:
        default_config_path = Path(__file__).parent / "configs" / "iac_che_config.yaml"
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
    train_split = config.get("dataset.train_split") or config.get(
        "dataset.split", "train"
    )
    eval_split = config.get("dataset.eval_split")
    train_size = config.get("dataset.size")
    eval_size = config.get("dataset.eval_size")
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
    # IAC-specific config (needed early for seed)
    # ------------------------------------------------------------------ #
    iac_cfg = config.get_section("iac") if hasattr(config, "get_section") else {}
    seed_value = int(config.get("seed", iac_cfg.get("seed", 42)))

    # ------------------------------------------------------------------ #
    # Output directory handling
    # ------------------------------------------------------------------ #
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")
    output_dir = os.path.join(output_base_dir, f"iac_job_{slurm_job_id}")
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "config.yaml")

    # ------------------------------------------------------------------ #
    # Tokenizer / dataset
    # ------------------------------------------------------------------ #
    _set_seed(seed_value)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, **model_config.tokenizer_kwargs
    )
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

    # ------------------------------------------------------------------ #
    # External context resolver (for multi-turn transitions)
    # ------------------------------------------------------------------ #
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

    if train_dataset is not None:
        _register_split(train_dataset)
    if eval_dataset is not None:
        _register_split(eval_dataset)

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    # Propagate verbosity to reward modules
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

    formatters = build_prompt_formatters()
    prompt_lookup = build_prompt_lookup(train_dataset)
    if eval_dataset is not None:
        prompt_lookup.update(build_prompt_lookup(eval_dataset))
    reward_fn = make_prompt_reward_fn(prompt_lookup)

    reward_processor = None
    shift_val = iac_cfg.get("reward_shift", -4)
    if shift_val is not None:
        try:
            shift_val_f = float(shift_val)
        except (TypeError, ValueError):
            shift_val_f = None
        if shift_val_f is not None:
            reward_processor = RewardProcessors.shift(value=shift_val_f)

    # ------------------------------------------------------------------ #
    # IAC-specific config
    # ------------------------------------------------------------------ #
    if "do_sample" in iac_cfg:
        use_sampling = bool(iac_cfg.get("do_sample"))
    else:
        use_sampling = bool(
            "temperature" in iac_cfg or "top_p" in iac_cfg or "top_k" in iac_cfg
        )
    top_k = iac_cfg.get("top_k")
    temperature = iac_cfg.get("temperature", 0.6)
    top_p = iac_cfg.get("top_p", 0.6)
    use_separate_critic = bool(iac_cfg.get("use_separate_critic", True))
    critic_model = (
        iac_cfg.get("critic_model") or iac_cfg.get("critic_model_name_or_path")
    )
    num_turns = iac_cfg.get("num_turns", 1)

    rollout_buffer_size = iac_cfg.get("rollout_buffer_size", 8)

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

    trainer = IACTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        reward_processor=reward_processor,
        formatters=formatters,
        metrics_callback=None,
        external_transition=external_transition_fn,
        args=IACConfig(
            output_dir=os.path.join(output_dir, "iac"),
            actor_learning_rate=iac_cfg.get("actor_learning_rate", 5e-6),
            critic_learning_rate=iac_cfg.get("critic_learning_rate", 5e-6),
            value_loss_coef=iac_cfg.get("value_loss_coef", 0.6),
            rollout_buffer_size=rollout_buffer_size,
            max_new_tokens=iac_cfg.get("max_new_tokens", 256),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_train_epochs=iac_cfg.get("num_train_epochs", 40),
            per_device_train_batch_size=iac_cfg.get("per_device_train_batch_size", 1),
            num_agents=iac_cfg.get("num_agents", 2),
            num_return_sequences=iac_cfg.get("num_return_sequences", 1),
            use_separate_critic=use_separate_critic,
            critic_model_name_or_path=critic_model,
            critic_value_head_hidden_dim=iac_cfg.get("critic_value_head_hidden_dim"),
            value_head_hidden_dim=iac_cfg.get("value_head_hidden_dim"),
            value_clip_range=iac_cfg.get("value_clip_range", 0.2),
            entropy_coef=iac_cfg.get("entropy_coef", 0.0),
            num_turns=num_turns,
            discount=iac_cfg.get("discount", 0.9),
            eval_interval=iac_cfg.get("eval_interval", 16),
            eval_num_samples=iac_cfg.get("eval_num_samples", 4),
            early_termination_threshold=iac_cfg.get(
                "early_termination_threshold", -0.2
            ),
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_config={
            "tokenizer_kwargs": model_config.tokenizer_kwargs,
            "model_kwargs": model_config.model_kwargs,
            "critic_model_kwargs": iac_cfg.get(
                "critic_model_kwargs", model_config.model_kwargs
            ),
        },
        wandb_config=_build_wandb_config(
            config, dataset_name, train_split, eval_split, train_size, eval_size
        ),
    )
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
    train_split: str,
    eval_split: str,
    train_size: int,
    eval_size: int | None,
):
    wandb_section = config.get_section("wandb") if hasattr(config, "get_section") else {}
    iac_section = config.get_section("iac") if hasattr(config, "get_section") else {}
    output_section = (
        config.get_section("output") if hasattr(config, "get_section") else {}
    )
    tags = wandb_section.get("tags", ["iac", dataset_name or "code", "turns_1"])
    return {
        "project": wandb_section.get("project", "iac"),
        "entity": wandb_section.get("entity"),
        "name": wandb_section.get("name", "iac_run"),
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
                "num_turns": iac_section.get("num_turns", 1),
                "max_new_tokens": iac_section.get("max_new_tokens", 256),
                "temperature": iac_section.get("temperature", 0.6),
                "top_p": iac_section.get("top_p", 0.6),
                "top_k": iac_section.get("top_k"),
                "use_separate_critic": iac_section.get(
                    "use_separate_critic", False
                ),
            },
        },
    }


if __name__ == "__main__":
    main()
