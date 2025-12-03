import argparse
import random
import re
from typing import Dict, List

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from rewards.code_rewards import execution_reward_aux


def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", prompt_text)
    if match:
        params_str = match.group(1)
        return [p.strip() for p in params_str.split(",") if p.strip()]
    return []


def aux_function_formatter(example: Dict[str, str]) -> str:
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)
    if not params or not entry_point:
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

def aux(...):
    # your function code here
    return result
"""


def main_function_formatter(example: Dict[str, str]) -> str:
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")

    params = extract_function_params_from_prompt(prompt)
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

def {entry_point}({params_str}):
    # your function code here
    return result
"""


def build_prompt_formatters() -> List:
    return [aux_function_formatter, main_function_formatter]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-turn MAGRPO baseline on CoopHumanEval code generation."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--dataset-name", type=str, default="CoMLRL/CoopHumanEval")
    parser.add_argument("--dataset-split", type=str, default="test[16:]")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./magrpo_che")
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--wandb-project", type=str, default="code-gen-compare")
    parser.add_argument("--wandb-entity", type=str, default="openmlrl")
    parser.add_argument("--wandb-run-name", type=str, default="codegen-magrpo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch_reward_fn():
    def _reward(
        aux_outputs: List[str],
        main_outputs: List[str],
        batch_items=None,
    ) -> List[float]:
        count = min(len(aux_outputs), len(main_outputs))
        if count == 0:
            return []
        if not batch_items:
            raise ValueError("batch_items must be provided for reward computation.")

        item = batch_items[0]
        test_code = item.get("test", "")
        entry_point = item.get("entry_point", "")
        raw_prompt = item.get("prompt", "")

        return execution_reward_aux(
            aux_outputs[:count],
            main_outputs[:count],
            [test_code] * count,
            [entry_point] * count,
            [raw_prompt] * count,
        )

    return _reward


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    usable = min(args.dataset_size, len(dataset))
    dataset = dataset.select(range(usable))

    magrpo_args = MAGRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_interval=0,
        num_turns=1,
    )

    trainer = MAGRPOTrainer(
        model=args.model_name,
        num_agents=2,
        tokenizer=tokenizer,
        reward_func=make_batch_reward_fn(),
        formatters=build_prompt_formatters(),
        args=magrpo_args,
        train_dataset=dataset,
        model_config={
            "tokenizer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {"trust_remote_code": True, "torch_dtype": "bfloat16"},
        },
        wandb_config={
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "config_sections": {
                "dataset": {
                    "name": args.dataset_name,
                    "split": args.dataset_split,
                    "size": usable,
                },
                "trainer": {
                    "num_generations": args.num_generations,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            },
        },
    )

    trainer.train()
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
