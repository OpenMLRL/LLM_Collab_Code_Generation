import argparse
import json
import os
import random
import re
import sys
from collections.abc import Sequence
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
    temperature = config.get(
        "preference.temperature", config.get("agent_model.temperature")
    )
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


def _generate_local(
    tokenizer,
    model,
    prompt: str,
    device: str,
    generation_kwargs: Dict[str, Any],
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    completion = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(completion, skip_special_tokens=True)[0]


def _generate_api(prompt: str, config: Config) -> str:
    provider = str(config.get("preference.comparator.provider", "")).lower()
    model = config.get("preference.comparator.model")
    if not provider or not model:
        raise ValueError("API comparator requires preference.comparator.provider and .model.")
    max_tokens = int(config.get("preference.max_new_tokens", 256))
    temperature = float(config.get("preference.temperature", 0.3))

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv(config.get("preference.comparator.api_key_env", "ANTHROPIC_API_KEY")))
        response = client.messages.create(
            model=str(model),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    if provider in {"openai", "deepseek"}:
        from openai import OpenAI

        api_key_env = config.get(
            "preference.comparator.api_key_env",
            "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY",
        )
        base_url = config.get(
            "preference.comparator.base_url",
            "https://api.deepseek.com" if provider == "deepseek" else None,
        )
        client = OpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
        response = client.chat.completions.create(
            model=str(model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return str(response.choices[0].message.content)

    raise ValueError(f"Unsupported API comparator provider: {provider}")


def _agent_names(config: Config, key: str) -> List[str] | None:
    names = config.get(key)
    if names is None:
        return None
    if not isinstance(names, Sequence) or isinstance(names, str) or len(names) != 2:
        raise ValueError(f"{key} must be a list of exactly two model names.")
    return [str(name) for name in names]


def _policy_agent_names(config: Config) -> List[str]:
    names = _agent_names(config, "preference.policy_agents")
    if names is not None:
        return names
    names = config.get("agents")
    if names is not None:
        if not isinstance(names, Sequence) or isinstance(names, str) or len(names) != 2:
            raise ValueError("agents must be a list of exactly two model names.")
        return [str(name) for name in names]
    model_name = str(config.get("agent_model.name"))
    return [model_name, model_name]


def _load_agent_group(
    names: List[str],
    loaded: Dict[str, Any],
    device: str,
    torch_dtype: str | None,
):
    agents = []
    for name in names:
        if name not in loaded:
            loaded[name] = _load_agent(name, device, torch_dtype)
        agents.append(loaded[name])
    return agents


def _generate_joint_local(
    agents,
    aux_prompt: str,
    main_prompt: str,
    device: str,
    generation_kwargs: Dict[str, Any],
) -> Tuple[str, str]:
    return (
        _generate_local(agents[0][0], agents[0][1], aux_prompt, device, generation_kwargs),
        _generate_local(agents[1][0], agents[1][1], main_prompt, device, generation_kwargs),
    )


def _pair_preference(samples: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    left, right = samples
    if left["oracle_reward"] == right["oracle_reward"]:
        return None
    chosen, rejected = (
        (left, right) if left["oracle_reward"] > right["oracle_reward"] else (right, left)
    )
    return {
        "chosen": chosen["sample_id"],
        "rejected": rejected["sample_id"],
        "margin": chosen["oracle_reward"] - rejected["oracle_reward"],
    }


def _task_id(item: Dict[str, Any], idx: int) -> str:
    return str(item.get("task_id") or item.get("id") or idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oracle-labeled preference pairs.")
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
            f"preference_buffers/{config.get('dataset.type')}_p{num_passes}.jsonl",
        )
    )
    if output_path.exists() and not bool(config.get("preference.overwrite", True)):
        raise FileExistsError(
            f"{output_path} already exists. Set preference.overwrite=true to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = str(config.get("preference.device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = config.get("preference.torch_dtype", config.get("agent_model.torch_dtype"))
    policy_names = _policy_agent_names(config)
    comparator_type = str(config.get("preference.comparator.type", "self")).lower()
    if comparator_type not in {"self", "ref", "api"}:
        raise ValueError("preference.comparator.type must be one of: self, ref, api.")

    loaded = {}
    policy_agents = _load_agent_group(policy_names, loaded, device, torch_dtype)
    comparator_agents = None
    if comparator_type == "self":
        comparator_agents = policy_agents
    elif comparator_type == "ref":
        ref_names = _agent_names(config, "preference.comparator.agents") or _agent_names(
            config, "preference.reference_agents"
        )
        if ref_names is None:
            raise ValueError("ref comparator requires preference.comparator.agents.")
        comparator_agents = _load_agent_group(ref_names, loaded, device, torch_dtype)

    generation_kwargs = _generation_kwargs(config)
    total_pairs = 0
    total_samples = 0
    selected = dataset.select(range(num_sets))
    with output_path.open("w") as f:
        progress = tqdm(
            total=num_passes * num_sets,
            desc="collect preference pairs",
            dynamic_ncols=True,
        )
        for pass_idx in range(num_passes):
            for idx, item in enumerate(selected):
                progress.set_description(
                    f"pass {pass_idx + 1}/{num_passes} task {idx + 1}/{num_sets}"
                )
                aux_prompt = _aux_function_formatter(item)
                main_prompt = _main_function_formatter(item)

                policy_aux, policy_main = _generate_joint_local(
                    policy_agents, aux_prompt, main_prompt, device, generation_kwargs
                )
                if comparator_type == "api":
                    comparator_aux = _generate_api(aux_prompt, config)
                    comparator_main = _generate_api(main_prompt, config)
                else:
                    comparator_aux, comparator_main = _generate_joint_local(
                        comparator_agents,
                        aux_prompt,
                        main_prompt,
                        device,
                        generation_kwargs,
                    )

                aux_outputs = [policy_aux, comparator_aux]
                main_outputs = [policy_main, comparator_main]
                rewards = execution_reward_aux(
                    aux_outputs,
                    main_outputs,
                    [item.get("test", "")] * 2,
                    [item.get("entry_point", "")] * 2,
                    [item.get("prompt", "")] * 2,
                )
                joint_samples = [
                    {
                        "sample_id": 0,
                        "source": "policy",
                        "aux": policy_aux,
                        "main": policy_main,
                        "oracle_reward": float(rewards[0]),
                    },
                    {
                        "sample_id": 1,
                        "source": "comparator",
                        "aux": comparator_aux,
                        "main": comparator_main,
                        "oracle_reward": float(rewards[1]),
                    },
                ]
                pair_preference = _pair_preference(joint_samples)
                record = {
                    "task_id": _task_id(item, idx),
                    "pass_id": pass_idx,
                    "prompt": item.get("prompt", ""),
                    "entry_point": item.get("entry_point", ""),
                    "test": item.get("test", ""),
                    "policy_agents": policy_names,
                    "comparator_type": comparator_type,
                    "joint_samples": joint_samples,
                    "pair_preference": pair_preference,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                total_pairs += int(pair_preference is not None)
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
        "joint_samples": total_samples,
        "pair_preferences": total_pairs,
        "comparator_type": comparator_type,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
