import argparse
import random
import re
from typing import Dict, List

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.maac import MAACConfig, MAACTrainer
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
        description="MAAC (shared critic) baseline on CoopHumanEval code generation."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument("--critic-model", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="CoMLRL/CoopHumanEval")
    parser.add_argument("--dataset-split", type=str, default="test[16:]")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./maac_che")
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=5e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-6)
    parser.add_argument("--value-loss-coef", type=float, default=0.6)
    parser.add_argument("--rollout-buffer-size", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default="code-gen-compare")
    parser.add_argument("--wandb-entity", type=str, default="openmlrl")
    parser.add_argument("--wandb-run-name", type=str, default="codegen-maac")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    formatters = build_prompt_formatters()
    prompt_lookup = build_prompt_lookup(dataset)
    reward_fn = make_prompt_reward_fn(prompt_lookup)

    use_sampling = args.num_generations > 1
    top_k = args.top_k if use_sampling else None

    trainer = MAACTrainer(
        model=args.model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=formatters,
        metrics_callback=None,
        args=MAACConfig(
            output_dir=f"{args.output_dir}/maac",
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            value_loss_coef=args.value_loss_coef,
            rollout_buffer_size=args.rollout_buffer_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_train_epochs=args.num_train_epochs,
            num_agents=2,
            num_return_sequences=args.num_generations,
            critic_model_name_or_path=args.critic_model or args.model_name,
        ),
        train_dataset=dataset,
        model_config={
            "tokenizer_kwargs": {"trust_remote_code": True},
            "model_kwargs": {"trust_remote_code": True, "torch_dtype": "bfloat16"},
            "critic_model_kwargs": {
                "trust_remote_code": True,
                "torch_dtype": "bfloat16",
            },
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
                    "top_k": args.top_k,
                },
            },
        },
    )
    trainer.train()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
