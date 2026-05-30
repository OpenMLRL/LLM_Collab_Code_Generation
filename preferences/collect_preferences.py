import argparse
import json
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config, add_config_args, parse_overrides
import rewards.code_rewards as code_rewards
from rewards.code_rewards import execution_reward_aux


def _torch_dtype(name: str | None):
    return getattr(torch, name) if name else None


def _load_agent(name: str, device: str, torch_dtype: str | None):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    dtype = _torch_dtype(torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs).to(device)
    model.eval()
    return tokenizer, model


def _extract_function_params(prompt_text: str) -> List[str]:
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if not match:
        return []
    return [p.strip() for p in match.group(1).split(",") if p.strip()]


def _aux_function_formatter(example: Dict[str, Any]) -> str:
    prompt = example.get("prompt", "")
    if not _extract_function_params(prompt) or not example.get("entry_point", ""):
        return "Error: Could not extract function information from prompt."
    return f"""Create a helper function for this coding problem.

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


def _main_function_formatter(example: Dict[str, Any]) -> str:
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    params = _extract_function_params(prompt)
    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."
    params_str = ", ".join(params)
    return f"""Solve this coding problem by implementing the required function.

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


def _generation_kwargs(config: Config) -> Dict[str, Any]:
    max_new_tokens = int(config.get("preference.max_new_tokens", 256))
    temperature = config.get("preference.temperature", config.get("agent_model.temperature"))
    top_p = config.get("preference.top_p", config.get("agent_model.top_p"))
    top_k = config.get("preference.top_k", config.get("agent_model.top_k"))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    return kwargs


def _generate(
    tokenizer,
    model,
    prompt: str,
    count: int,
    device: str,
    generation_kwargs: Dict[str, Any],
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        num_return_sequences=count,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    completions = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)


def _preference_views(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    scores = [
        {"sample_id": sample["sample_id"], "score": sample["oracle_reward"]}
        for sample in samples
    ]

    pairs = []
    by_id = {sample["sample_id"]: sample for sample in samples}
    for left_id, right_id in combinations(by_id.keys(), 2):
        left = by_id[left_id]
        right = by_id[right_id]
        if left["oracle_reward"] == right["oracle_reward"]:
            continue
        chosen, rejected = (
            (left, right)
            if left["oracle_reward"] > right["oracle_reward"]
            else (right, left)
        )
        pairs.append(
            {
                "chosen": chosen["sample_id"],
                "rejected": rejected["sample_id"],
                "margin": chosen["oracle_reward"] - rejected["oracle_reward"],
            }
        )

    ranking = [
        {"sample_id": sample["sample_id"], "score": sample["oracle_reward"]}
        for sample in sorted(samples, key=lambda x: (-x["oracle_reward"], x["sample_id"]))
    ]
    return scores, pairs, ranking


def _task_id(item: Dict[str, Any], idx: int) -> str:
    return str(item.get("task_id") or item.get("id") or idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oracle-labeled preference buffers.")
    add_config_args(parser)
    args = parser.parse_args()

    if not args.config:
        raise ValueError("Please provide --config.")
    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))
    code_rewards.VERBOSE = bool(config.get("output.verbose", False))

    seed = int(config.get("preference.seed", config.get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)

    set_size = int(config.get("preference.set_size", 2))
    if set_size < 2:
        raise ValueError("preference.set_size must be >= 2.")
    num_passes = int(config.get("preference.num_passes", 1))
    if num_passes < 1:
        raise ValueError("preference.num_passes must be >= 1.")

    split = config.get("preference.split", config.get("dataset.train_split"))
    dataset = load_dataset(config.get("dataset.name"), split=split)
    num_sets = int(config.get("preference.num_sets", len(dataset)))
    if num_sets > len(dataset):
        raise ValueError("preference.num_sets cannot exceed the selected dataset split size.")
    if bool(config.get("preference.shuffle", True)):
        dataset = dataset.shuffle(seed=seed)

    output_path = Path(
        config.get(
            "preference.output_path",
            f"preference_buffers/{config.get('dataset.type')}_s{set_size}.jsonl",
        )
    )
    if output_path.exists() and not bool(config.get("preference.overwrite", True)):
        raise FileExistsError(
            f"{output_path} already exists. Set preference.overwrite=true to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = str(config.get("preference.device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = config.get("preference.torch_dtype", config.get("agent_model.torch_dtype"))
    agent_names = config.get("agents") or [
        config.get("agent_model.name"),
        config.get("agent_model.name"),
    ]
    if len(agent_names) != 2:
        raise ValueError("preference collection expects exactly two agents.")

    loaded = {}
    agents = []
    for name in agent_names:
        if name not in loaded:
            loaded[name] = _load_agent(name, device, torch_dtype)
        agents.append(loaded[name])

    generation_kwargs = _generation_kwargs(config)
    total_pairs = 0
    total_samples = 0
    with output_path.open("w") as f:
        selected = dataset.select(range(num_sets))
        progress = tqdm(
            total=num_passes * num_sets,
            desc="collect preferences",
            dynamic_ncols=True,
        )
        for pass_idx in range(num_passes):
            for idx, item in enumerate(selected):
                progress.set_description(
                    f"pass {pass_idx + 1}/{num_passes} task {idx + 1}/{num_sets}"
                )
                aux_prompt = _aux_function_formatter(item)
                main_prompt = _main_function_formatter(item)

                aux_outputs = _generate(
                    agents[0][0],
                    agents[0][1],
                    aux_prompt,
                    set_size,
                    device,
                    generation_kwargs,
                )
                main_outputs = _generate(
                    agents[1][0],
                    agents[1][1],
                    main_prompt,
                    set_size,
                    device,
                    generation_kwargs,
                )

                rewards = execution_reward_aux(
                    aux_outputs,
                    main_outputs,
                    [item.get("test", "")] * set_size,
                    [item.get("entry_point", "")] * set_size,
                    [item.get("prompt", "")] * set_size,
                )
                joint_samples = [
                    {
                        "sample_id": sample_id,
                        "aux": aux_outputs[sample_id],
                        "main": main_outputs[sample_id],
                        "oracle_reward": float(rewards[sample_id]),
                    }
                    for sample_id in range(set_size)
                ]
                score_preferences, pair_preferences, ranking_preferences = (
                    _preference_views(joint_samples)
                )
                record = {
                    "task_id": _task_id(item, idx),
                    "pass_id": pass_idx,
                    "prompt": item.get("prompt", ""),
                    "entry_point": item.get("entry_point", ""),
                    "test": item.get("test", ""),
                    "joint_samples": joint_samples,
                    "score_preferences": score_preferences,
                    "pair_preferences": pair_preferences,
                    "ranking_preferences": ranking_preferences,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                total_pairs += len(pair_preferences)
                total_samples += len(joint_samples)
                progress.update(1)
                progress.set_postfix(
                    joint_samples=total_samples,
                    pair_preferences=total_pairs,
                    refresh=False,
                )
        progress.close()

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary = {
        "buffer_path": str(output_path),
        "dataset": config.get("dataset.name"),
        "split": split,
        "num_sets": num_sets,
        "num_passes": num_passes,
        "set_size": set_size,
        "joint_samples": total_samples,
        "pair_preferences": total_pairs,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
