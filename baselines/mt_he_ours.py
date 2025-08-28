import argparse
import json
import os
import re
import time
from collections import defaultdict
from math import comb
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from anthropic import Anthropic
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the logger functions
from loggers.code_logger import (
    aggregate_code_metrics_for_logging,
    code_reward_logger,
)
from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)

# Import external module for expert feedback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external import get_expert_feedback


class TwoTurnHumanEvalEvaluator:
    def __init__(
        self, aux_model_name, main_model_name, expert_model_name, device="auto"
    ):
        """Initialize the two-turn evaluator for HumanEval."""
        self.aux_model_name = aux_model_name
        self.main_model_name = main_model_name
        self.expert_model_name = expert_model_name
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading auxiliary model {aux_model_name} on {self.device}...")
        self.aux_tokenizer = AutoTokenizer.from_pretrained(aux_model_name)
        self.aux_model = AutoModelForCausalLM.from_pretrained(
            aux_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        print(f"Loading main model {main_model_name} on {self.device}...")
        self.main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.aux_model = self.aux_model.to(self.device)
            self.main_model = self.main_model.to(self.device)

        self.aux_model.eval()
        self.main_model.eval()

        # Add pad tokens if not present
        if self.aux_tokenizer.pad_token is None:
            self.aux_tokenizer.pad_token = self.aux_tokenizer.eos_token
        if self.main_tokenizer.pad_token is None:
            self.main_tokenizer.pad_token = self.main_tokenizer.eos_token

    def extract_function_params_from_prompt(self, prompt_text):
        """Extract function parameters from the prompt text."""
        match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
        if match:
            params_str = match.group(1)
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            return params
        return []

    def aux_function_formatter(
        self, prompt: str, entry_point: str, expert_feedback: Optional[str] = None
    ) -> str:
        """Formatter for the auxiliary function generator with optional expert feedback."""
        params = self.extract_function_params_from_prompt(prompt)
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

def aux({params_str}):
    # your function code here
    return result
"""

        if expert_feedback is not None:
            prompt_text += (
                f"\n\nHere is the feedback from an expert:\n{expert_feedback}"
            )

        return prompt_text

    def main_function_formatter(
        self, prompt: str, entry_point: str, expert_feedback: Optional[str] = None
    ) -> str:
        """Formatter for the main function generator with optional expert feedback."""
        params = self.extract_function_params_from_prompt(prompt)
        if not params or not entry_point:
            return "Error: Could not extract function information from prompt."

        params_str = ", ".join(params)

        prompt_text = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux({params_str})

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

        if expert_feedback is not None:
            prompt_text += (
                f"\n\nHere is the feedback from an expert:\n{expert_feedback}"
            )

        return prompt_text

    def generate_response(
        self, model, tokenizer, prompt: str, max_new_tokens: int = 256
    ) -> tuple:
        """Generate response for a given prompt using specified model and tokenizer."""
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                # Add optimization parameters
                use_cache=True,
                num_beams=1,  # Greedy decoding is faster
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_tokens = len(generated_tokens)
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip(), input_tokens, output_tokens

    def generate_batch_responses(
        self,
        model,
        tokenizer,
        prompts: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> List[tuple]:
        """Generate responses for multiple prompts in batches."""
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                )

            # Decode each response
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                input_tokens = input_length
                output_tokens = len(generated_tokens)

                results.append((response.strip(), input_tokens, output_tokens))

        return results

    def calculate_new_metrics_per_sample(self, sample_metrics: list, k: int) -> dict:
        """
        Calculate metrics (PassOrNot@k, Accuracy@k, MaxCoop@k) for a single sample.
        For multi-turn, this works on the final turn (Turn 2) metrics.

        Args:
            sample_metrics: List of metrics for all generations of this sample (final turn only)
            k: Number of top generations to consider

        Returns:
            dict: Contains passornot_at_k, accuracy_at_k, maxcoop_at_k
        """
        if not sample_metrics or k <= 0:
            return {"passornot_at_k": 0.0, "accuracy_at_k": 0.0, "maxcoop_at_k": 0.0}

        # Take first k generations (or all if k > number of generations)
        k_metrics = sample_metrics[: min(k, len(sample_metrics))]

        # 1. PassOrNot@k: Check if any of the k solutions passed ALL tests
        passornot_at_k = 0.0
        for metric in k_metrics:
            # A solution passes all tests if passed_tests == total_tests and total_tests > 0
            if metric.get("total_tests", 0) > 0 and metric.get(
                "passed_tests", 0
            ) == metric.get("total_tests", 0):
                passornot_at_k = 1.0
                break

        # 2. Accuracy@k: Find the solution with highest test pass rate among k solutions
        accuracy_at_k = 0.0
        for metric in k_metrics:
            if metric.get("total_tests", 0) > 0:
                current_accuracy = metric.get("passed_tests", 0) / metric.get(
                    "total_tests", 0
                )
                accuracy_at_k = max(accuracy_at_k, current_accuracy)

        # 3. MaxCoop@k: Find the solution with highest cooperation score among k solutions
        maxcoop_at_k = 0.0
        for metric in k_metrics:
            maxcoop_at_k = max(maxcoop_at_k, metric.get("bonus_reward", 0.0))

        return {
            "passornot_at_k": passornot_at_k,
            "accuracy_at_k": accuracy_at_k,
            "maxcoop_at_k": maxcoop_at_k,
        }

    def evaluate_two_turn_humaneval(
        self,
        num_samples: int = 31,
        num_generations: int = 1,
        k_values: list = [1, 5, 10],
        save_results: bool = True,
        skip_expert_feedback: bool = False,  # Add option to skip expert feedback
    ):
        """Evaluate HumanEval with two-turn system including expert feedback."""
        print(f"Loading HumanEval dataset...")
        dataset_name = "openai/openai_humaneval"
        test_data = load_dataset(dataset_name, split="test[133:]")

        # Take the last num_samples
        total_samples = len(test_data)
        start_idx = max(0, total_samples - num_samples)
        test_samples = test_data.select(range(start_idx, total_samples))

        print(
            f"Evaluating on samples {start_idx} to {total_samples - 1} ({len(test_samples)} samples)..."
        )
        print(f"Generating {num_generations} completions per sample...")
        if skip_expert_feedback:
            print(
                "Running 2-turn evaluation WITHOUT expert feedback (faster testing mode)..."
            )
        else:
            print("Running 2-turn evaluation with expert feedback between turns...")

        # Store results per sample
        sample_results = []
        all_response_times = []
        all_input_tokens = []
        all_output_tokens = []
        failed_generations = 0

        # Prepare data for mt_humaneval_logger
        all_completions1_turns = []  # [sample][turn]
        all_completions2_turns = []  # [sample][turn]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []

        for sample_idx, sample in enumerate(
            tqdm(test_samples, desc="Processing samples")
        ):
            prompt = sample["prompt"]
            test_code = sample["test"]
            entry_point = sample["entry_point"]

            sample_data = {
                "sample_idx": sample_idx,
                "entry_point": entry_point,
                "prompt": prompt,
                "test_code": test_code,
                "generations": [],  # Store all generation attempts
            }

            # Time tracking for ETA
            sample_start_time = time.time()

            # For each generation of this sample
            for gen_idx in range(num_generations):
                if gen_idx > 0:
                    avg_gen_time = (time.time() - sample_start_time) / gen_idx
                    eta = avg_gen_time * (num_generations - gen_idx)
                    print(
                        f"\n  Sample {sample_idx+1}/{len(test_samples)}, Generation {gen_idx+1}/{num_generations}, ETA: {eta:.1f}s"
                    )

                generation_data = {
                    "gen_idx": gen_idx,
                    "completions1_turns": [],  # aux completions for each turn
                    "completions2_turns": [],  # main completions for each turn
                    "generation_times": [],
                    "input_tokens": [],
                    "output_tokens": [],
                }

                try:
                    # Turn 1: Initial generation without expert feedback
                    # Generate auxiliary function
                    aux_prompt = self.aux_function_formatter(prompt, entry_point)
                    start_time = time.time()
                    aux_response, aux_input_tokens, aux_output_tokens = (
                        self.generate_response(
                            self.aux_model, self.aux_tokenizer, aux_prompt
                        )
                    )
                    end_time = time.time()
                    agent1_time = end_time - start_time

                    # Generate main function
                    main_prompt = self.main_function_formatter(prompt, entry_point)
                    start_time = time.time()
                    main_response, main_input_tokens, main_output_tokens = (
                        self.generate_response(
                            self.main_model, self.main_tokenizer, main_prompt
                        )
                    )
                    end_time = time.time()
                    agent2_time = end_time - start_time

                    # Store turn 1 completions
                    generation_data["completions1_turns"].append(aux_response)
                    generation_data["completions2_turns"].append(main_response)
                    generation_data["generation_times"].append(
                        (agent1_time + agent2_time) / 2
                    )
                    generation_data["input_tokens"].append(
                        aux_input_tokens + main_input_tokens
                    )
                    generation_data["output_tokens"].append(
                        aux_output_tokens + main_output_tokens
                    )

                    # Always do Turn 2 - let mt_humaneval_logger handle early termination
                    # Turn 2: Generation with expert feedback
                    if skip_expert_feedback:
                        # Skip expert feedback - use same responses for turn 2
                        aux_response2 = aux_response
                        main_response2 = main_response
                        agent1_time2 = 0
                        agent2_time2 = 0
                        aux_input_tokens2 = 0
                        aux_output_tokens2 = 0
                        main_input_tokens2 = 0
                        main_output_tokens2 = 0
                    else:
                        # Get expert feedback based on Turn 1 code
                        aux_expert_feedback, main_expert_feedback = get_expert_feedback(
                            prompt=prompt,
                            test=test_code,
                            best_reward=2.0,  # Default middle value since we don't evaluate yet
                            aux_completion=aux_response,
                            main_completion=main_response,
                            expert_model=self.expert_model_name,
                        )

                        # Generate with expert feedback
                        aux_prompt_with_feedback = self.aux_function_formatter(
                            prompt, entry_point, expert_feedback=aux_expert_feedback
                        )
                        start_time = time.time()
                        aux_response2, aux_input_tokens2, aux_output_tokens2 = (
                            self.generate_response(
                                self.aux_model,
                                self.aux_tokenizer,
                                aux_prompt_with_feedback,
                            )
                        )
                        end_time = time.time()
                        agent1_time2 = end_time - start_time

                        main_prompt_with_feedback = self.main_function_formatter(
                            prompt, entry_point, expert_feedback=main_expert_feedback
                        )
                        start_time = time.time()
                        main_response2, main_input_tokens2, main_output_tokens2 = (
                            self.generate_response(
                                self.main_model,
                                self.main_tokenizer,
                                main_prompt_with_feedback,
                            )
                        )
                        end_time = time.time()
                        agent2_time2 = end_time - start_time

                    # Store turn 2 completions
                    generation_data["completions1_turns"].append(aux_response2)
                    generation_data["completions2_turns"].append(main_response2)
                    generation_data["generation_times"].append(
                        (agent1_time2 + agent2_time2) / 2
                    )
                    generation_data["input_tokens"].append(
                        aux_input_tokens2 + main_input_tokens2
                    )
                    generation_data["output_tokens"].append(
                        aux_output_tokens2 + main_output_tokens2
                    )

                    # Track timing and tokens
                    total_time = sum(generation_data["generation_times"])
                    total_input_tokens = sum(generation_data["input_tokens"])
                    total_output_tokens = sum(generation_data["output_tokens"])

                    all_response_times.append(total_time)
                    all_input_tokens.append(total_input_tokens)
                    all_output_tokens.append(total_output_tokens)

                except Exception as e:
                    print(f"Failed on sample {sample_idx}, generation {gen_idx}: {e}")
                    failed_generations += 1
                    generation_data["error"] = str(e)

                sample_data["generations"].append(generation_data)

            # Prepare data for mt_humaneval_logger
            # We need to organize completions by generation
            for gen_idx in range(num_generations):
                if (
                    gen_idx < len(sample_data["generations"])
                    and "completions1_turns" in sample_data["generations"][gen_idx]
                ):
                    all_completions1_turns.append(
                        sample_data["generations"][gen_idx]["completions1_turns"]
                    )
                    all_completions2_turns.append(
                        sample_data["generations"][gen_idx]["completions2_turns"]
                    )
                    all_test_cases.append(test_code)
                    all_entry_points.append(entry_point)
                    all_prompts.append(prompt)

            sample_results.append(sample_data)

        print(f"Generation complete. Failed generations: {failed_generations}")

        # Use mt_humaneval_logger to get metrics for all samples
        print("Calculating metrics using mt_humaneval_logger...")
        mt_metrics = mt_humaneval_logger(
            all_completions1_turns,
            all_completions2_turns,
            all_test_cases,
            all_entry_points,
            all_prompts,
        )

        # Count early terminations from mt_metrics
        early_terminations = sum(
            1 for m in mt_metrics if m.get("early_termination", False)
        )
        print(f"Early terminations: {early_terminations}")

        # Aggregate metrics using the logger's aggregation function
        aggregated_mt_metrics = aggregate_mt_humaneval_metrics_for_logging(
            mt_metrics, num_turns=2
        )

        # Calculate pass@k and other metrics
        aggregated_metrics = self._calculate_aggregated_metrics(
            sample_results,
            mt_metrics,
            k_values,
            all_response_times,
            all_input_tokens,
            all_output_tokens,
            failed_generations,
            aggregated_mt_metrics,
        )

        # Print results
        self._print_results(
            aggregated_metrics, len(test_samples), start_idx, total_samples
        )

        # Save detailed results
        if save_results:
            results = {
                "aggregated_metrics": aggregated_metrics,
                "sample_results": sample_results,
                "mt_metrics": mt_metrics,  # Include raw metrics from logger
                "evaluation_config": {
                    "num_samples": num_samples,
                    "num_generations": num_generations,
                    "k_values": k_values,
                    "sample_indices": f"{start_idx}-{total_samples - 1}",
                    "num_turns": 2,
                    "dataset": "openai/openai_humaneval",
                },
            }

            filename = f"humaneval_2turn_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, sample_results

    def _calculate_aggregated_metrics(
        self,
        sample_results,
        mt_metrics,
        k_values,
        all_response_times,
        all_input_tokens,
        all_output_tokens,
        failed_generations,
        aggregated_mt_metrics,
    ):
        """Calculate aggregated metrics across all samples."""
        early_termination_count = sum(
            1 for sample in mt_metrics if sample.get("early_termination", False)
        )

        total_generations = sum(len(s["generations"]) for s in sample_results)

        # Calculate NEW metrics for Turn 1 and Turn 2
        turn1_pass_at_k = self._calculate_new_pass_at_k_for_turn(
            sample_results, mt_metrics, k_values, turn_idx=0
        )
        turn2_pass_at_k = self._calculate_new_pass_at_k_for_turn(
            sample_results, mt_metrics, k_values, turn_idx=1
        )

        # Extract turn-specific averages from aggregated_mt_metrics
        turn1_avg_metrics = {
            "avg_total_reward": aggregated_mt_metrics.get("turn_1/avg_total_reward", 0),
            "avg_level_1_reward": aggregated_mt_metrics.get(
                "turn_1/avg_level_1_reward", 0
            ),
            "avg_level_2_reward": aggregated_mt_metrics.get(
                "turn_1/avg_level_2_reward", 0
            ),
            "avg_level_3_reward": aggregated_mt_metrics.get(
                "turn_1/avg_level_3_reward", 0
            ),
            "avg_test_reward": aggregated_mt_metrics.get("turn_1/avg_test_reward", 0),
            "avg_bonus_reward": aggregated_mt_metrics.get("turn_1/avg_bonus_reward", 0),
            "structure_extraction_rate": (
                aggregated_mt_metrics.get("turn_1/avg_level_1_reward", 0) / 1.0
                if aggregated_mt_metrics.get("turn_1/avg_level_1_reward", 0) >= 0.6
                else 0
            ),
            "syntax_correct_rate": (
                1.0
                if aggregated_mt_metrics.get("turn_1/avg_level_2_reward", 0) > 0
                else 0
            ),
            "full_test_pass_rate": self._calculate_full_test_pass_rate(
                mt_metrics, turn_idx=0
            ),
            "cooperation_rate": self._calculate_cooperation_rate(
                mt_metrics, turn_idx=0
            ),
            "avg_passed_tests": aggregated_mt_metrics.get("turn_1/avg_passed_tests", 0),
            "avg_total_tests": aggregated_mt_metrics.get("turn_1/avg_total_tests", 0),
            "avg_timeout_num": aggregated_mt_metrics.get("turn_1/avg_timeout_num", 0),
        }

        turn2_avg_metrics = {
            "avg_total_reward": aggregated_mt_metrics.get("turn_2/avg_total_reward", 0),
            "avg_level_1_reward": aggregated_mt_metrics.get(
                "turn_2/avg_level_1_reward", 0
            ),
            "avg_level_2_reward": aggregated_mt_metrics.get(
                "turn_2/avg_level_2_reward", 0
            ),
            "avg_level_3_reward": aggregated_mt_metrics.get(
                "turn_2/avg_level_3_reward", 0
            ),
            "avg_test_reward": aggregated_mt_metrics.get("turn_2/avg_test_reward", 0),
            "avg_bonus_reward": aggregated_mt_metrics.get("turn_2/avg_bonus_reward", 0),
            "structure_extraction_rate": (
                aggregated_mt_metrics.get("turn_2/avg_level_1_reward", 0) / 1.0
                if aggregated_mt_metrics.get("turn_2/avg_level_1_reward", 0) >= 0.6
                else 0
            ),
            "syntax_correct_rate": (
                1.0
                if aggregated_mt_metrics.get("turn_2/avg_level_2_reward", 0) > 0
                else 0
            ),
            "full_test_pass_rate": self._calculate_full_test_pass_rate(
                mt_metrics, turn_idx=1
            ),
            "cooperation_rate": self._calculate_cooperation_rate(
                mt_metrics, turn_idx=1
            ),
            "avg_passed_tests": aggregated_mt_metrics.get("turn_2/avg_passed_tests", 0),
            "avg_total_tests": aggregated_mt_metrics.get("turn_2/avg_total_tests", 0),
            "avg_timeout_num": aggregated_mt_metrics.get("turn_2/avg_timeout_num", 0),
        }

        # Token and timing statistics
        avg_input_tokens = np.mean(all_input_tokens) if all_input_tokens else 0
        avg_output_tokens = np.mean(all_output_tokens) if all_output_tokens else 0
        avg_response_time = np.mean(all_response_times) if all_response_times else 0

        aggregated_metrics = {
            # Basic info
            "aux_model_name": self.aux_model_name,
            "main_model_name": self.main_model_name,
            "dataset": "openai/openai_humaneval",
            "num_samples": len(sample_results),
            "num_generations_per_sample": (
                len(sample_results[0]["generations"]) if sample_results else 0
            ),
            "total_generations": total_generations,
            "failed_generations": failed_generations,
            "early_termination_rate": (
                early_termination_count / total_generations
                if total_generations > 0
                else 0
            ),
            # Turn 1 metrics
            "turn1": {
                **turn1_avg_metrics,
                "pass_at_k": turn1_pass_at_k,
            },
            # Turn 2 metrics
            "turn2": {
                **turn2_avg_metrics,
                "pass_at_k": turn2_pass_at_k,
            },
            # Improvement metrics
            "avg_improvement": aggregated_mt_metrics.get("turn_2/avg_improvement", 0),
            "improvement_rate": self._calculate_improvement_rate(mt_metrics),
            "max_improvement": self._calculate_max_improvement(mt_metrics),
            "min_improvement": self._calculate_min_improvement(mt_metrics),
            # Token and timing metrics
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_input_tokens + avg_output_tokens,
            "avg_response_time": avg_response_time,
            "total_time": np.sum(all_response_times),
        }

        return aggregated_metrics

    def _calculate_new_pass_at_k_for_turn(
        self, sample_results, mt_metrics, k_values, turn_idx
    ):
        """
        Calculate NEW pass@k metrics for a specific turn using mt_metrics.
        """
        passornot_at_k = {}
        accuracy_at_k = {}
        maxcoop_at_k = {}

        for k in k_values:
            passornot_results = []
            accuracy_results = []
            maxcoop_results = []

            # Group metrics by sample
            sample_metrics_dict = defaultdict(list)
            for i, metric in enumerate(mt_metrics):
                sample_idx = (
                    i // len(sample_results[0]["generations"])
                    if sample_results and sample_results[0]["generations"]
                    else 0
                )
                sample_metrics_dict[sample_idx].append(metric)

            for sample_idx, sample_metrics_list in sample_metrics_dict.items():
                # Get metrics for this turn from the first k generations
                turn_metrics = []

                for gen_idx, metric in enumerate(sample_metrics_list[:k]):
                    turn_key = f"turn_{turn_idx + 1}"
                    turn_metric = {
                        "total_tests": metric.get(f"{turn_key}/total_tests", 0),
                        "passed_tests": metric.get(f"{turn_key}/passed_tests", 0),
                        "bonus_reward": metric.get(f"{turn_key}/bonus_reward", 0.0),
                    }
                    turn_metrics.append(turn_metric)

                # Calculate NEW metrics for this sample
                if turn_metrics:
                    new_metrics = self.calculate_new_metrics_per_sample(turn_metrics, k)
                    passornot_results.append(new_metrics["passornot_at_k"])
                    accuracy_results.append(new_metrics["accuracy_at_k"])
                    maxcoop_results.append(new_metrics["maxcoop_at_k"])

            # Average NEW metrics across all samples
            passornot_at_k[f"PassOrNot@{k}"] = (
                np.mean(passornot_results) if passornot_results else 0
            )
            accuracy_at_k[f"Accuracy@{k}"] = (
                np.mean(accuracy_results) if accuracy_results else 0
            )
            maxcoop_at_k[f"MaxCoop@{k}"] = (
                np.mean(maxcoop_results) if maxcoop_results else 0
            )

        return {
            "passornot": passornot_at_k,
            "accuracy": accuracy_at_k,
            "cooperation": maxcoop_at_k,
        }

    def _calculate_full_test_pass_rate(self, mt_metrics, turn_idx):
        """Calculate the rate of solutions that passed all tests for a specific turn."""
        full_pass_count = 0
        total_count = 0

        turn_key = f"turn_{turn_idx + 1}"
        for metric in mt_metrics:
            total_tests = metric.get(f"{turn_key}/total_tests", 0)
            passed_tests = metric.get(f"{turn_key}/passed_tests", 0)

            if total_tests > 0:
                total_count += 1
                if passed_tests == total_tests:
                    full_pass_count += 1

        return full_pass_count / total_count if total_count > 0 else 0

    def _calculate_cooperation_rate(self, mt_metrics, turn_idx):
        """Calculate the rate of solutions with positive cooperation bonus for a specific turn."""
        coop_count = 0
        total_count = 0

        turn_key = f"turn_{turn_idx + 1}"
        for metric in mt_metrics:
            bonus_reward = metric.get(f"{turn_key}/bonus_reward", 0)
            total_count += 1
            if bonus_reward > 0:
                coop_count += 1

        return coop_count / total_count if total_count > 0 else 0

    def _calculate_improvement_rate(self, mt_metrics):
        """Calculate the rate of solutions that improved from turn 1 to turn 2."""
        improved_count = 0
        total_count = 0

        for metric in mt_metrics:
            if not metric.get("early_termination", False):
                improvement = metric.get("turn_2/improvement_from_turn_1", 0)
                total_count += 1
                if improvement > 0:
                    improved_count += 1

        return improved_count / total_count if total_count > 0 else 0

    def _calculate_max_improvement(self, mt_metrics):
        """Calculate the maximum improvement from turn 1 to turn 2."""
        improvements = []
        for metric in mt_metrics:
            if not metric.get("early_termination", False):
                improvement = metric.get("turn_2/improvement_from_turn_1", 0)
                improvements.append(improvement)

        return max(improvements) if improvements else 0

    def _calculate_min_improvement(self, mt_metrics):
        """Calculate the minimum improvement from turn 1 to turn 2."""
        improvements = []
        for metric in mt_metrics:
            if not metric.get("early_termination", False):
                improvement = metric.get("turn_2/improvement_from_turn_1", 0)
                improvements.append(improvement)

        return min(improvements) if improvements else 0

    def _print_results(self, aggregated_metrics, num_samples, start_idx, total_samples):
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("TWO-TURN HUMANEVAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset: openai/openai_humaneval")
        print(f"Aux Model: {self.aux_model_name}")
        print(f"Main Model: {self.main_model_name}")
        print(
            f"Samples evaluated: {num_samples} (indices {start_idx}-{total_samples - 1})"
        )
        print(
            f"Generations per sample: {aggregated_metrics['num_generations_per_sample']}"
        )
        print(f"Total generations: {aggregated_metrics['total_generations']}")
        print(
            f"Early termination rate: {aggregated_metrics['early_termination_rate']:.1%}"
        )
        print()

        print("TIMING AND TOKEN METRICS:")
        print(f"Average response time: {aggregated_metrics['avg_response_time']:.2f}s")
        print(f"Total time: {aggregated_metrics['total_time']:.1f}s")
        print(f"Average input tokens: {aggregated_metrics['avg_input_tokens']:.1f}")
        print(f"Average output tokens: {aggregated_metrics['avg_output_tokens']:.1f}")
        print(f"Average total tokens: {aggregated_metrics['avg_total_tokens']:.1f}")
        print()

        # Turn 1 Results
        print("TURN 1 RESULTS:")
        print("-" * 30)
        turn1 = aggregated_metrics["turn1"]
        print(
            f"Structure extraction rate: {turn1.get('structure_extraction_rate', 0):.3f}"
        )
        print(f"Syntax correct rate: {turn1.get('syntax_correct_rate', 0):.3f}")
        print(f"Full test pass rate: {turn1.get('full_test_pass_rate', 0):.3f}")
        print(f"Cooperation rate: {turn1.get('cooperation_rate', 0):.3f}")
        print(f"Average total reward: {turn1.get('avg_total_reward', 0):.3f}")

        print("\nTurn 1 NEW METRICS:")
        print("PassOrNot@k (at least one solution passes ALL tests):")
        for k, v in turn1["pass_at_k"]["passornot"].items():
            print(f"  {k}: {v:.3f}")
        print("Accuracy@k (best test accuracy among k solutions):")
        for k, v in turn1["pass_at_k"]["accuracy"].items():
            print(f"  {k}: {v:.3f}")
        print("MaxCoop@k (best cooperation score among k solutions):")
        for k, v in turn1["pass_at_k"]["cooperation"].items():
            print(f"  {k}: {v:.3f}")
        print()

        # Turn 2 Results
        print("TURN 2 RESULTS (WITH EXPERT FEEDBACK):")
        print("-" * 30)
        turn2 = aggregated_metrics["turn2"]
        print(
            f"Structure extraction rate: {turn2.get('structure_extraction_rate', 0):.3f}"
        )
        print(f"Syntax correct rate: {turn2.get('syntax_correct_rate', 0):.3f}")
        print(f"Full test pass rate: {turn2.get('full_test_pass_rate', 0):.3f}")
        print(f"Cooperation rate: {turn2.get('cooperation_rate', 0):.3f}")
        print(f"Average total reward: {turn2.get('avg_total_reward', 0):.3f}")

        print("\nTurn 2 NEW METRICS:")
        print("PassOrNot@k (at least one solution passes ALL tests):")
        for k, v in turn2["pass_at_k"]["passornot"].items():
            print(f"  {k}: {v:.3f}")
        print("Accuracy@k (best test accuracy among k solutions):")
        for k, v in turn2["pass_at_k"]["accuracy"].items():
            print(f"  {k}: {v:.3f}")
        print("MaxCoop@k (best cooperation score among k solutions):")
        for k, v in turn2["pass_at_k"]["cooperation"].items():
            print(f"  {k}: {v:.3f}")
        print()

        # Improvement Metrics
        print("IMPROVEMENT METRICS (TURN 2 vs TURN 1):")
        print("-" * 30)
        print(f"Average improvement: {aggregated_metrics['avg_improvement']:.3f}")
        print(f"Improvement rate: {aggregated_metrics['improvement_rate']:.1%}")
        print(f"Max improvement: {aggregated_metrics['max_improvement']:.3f}")
        print(f"Min improvement: {aggregated_metrics['min_improvement']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Two-Turn HumanEval Evaluation")
    parser.add_argument(
        "--aux-model",
        default="LovelyBuggies/2xQwen2.5-Coder-3B-Cyclops-Aux",
        help="Auxiliary model name",
    )
    parser.add_argument(
        "--main-model",
        default="LovelyBuggies/2xQwen2.5-Coder-3B-Cyclops-Main",
        help="Main model name",
    )
    parser.add_argument(
        "--samples", type=int, default=31, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--generations", type=int, default=1, help="Number of generations per sample"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K values for pass@k calculation",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save detailed results"
    )
    parser.add_argument(
        "--expert-model",
        default="deepseek-coder",
        help="Expert model for feedback (claude-3-5-sonnet-20241022, deepseek-coder, qwen3-coder)",
    )
    parser.add_argument(
        "--skip-expert-feedback",
        action="store_true",
        help="Skip expert feedback for faster testing (uses Turn 1 results for Turn 2)",
    )

    args = parser.parse_args()

    # Initialize two-turn evaluator
    evaluator = TwoTurnHumanEvalEvaluator(
        aux_model_name=args.aux_model,
        main_model_name=args.main_model,
        expert_model_name=args.expert_model,
        device=args.device,
    )

    # Run evaluation
    aggregated_metrics, sample_results = evaluator.evaluate_two_turn_humaneval(
        num_samples=args.samples,
        num_generations=args.generations,
        k_values=args.k_values,
        save_results=not args.no_save,
        skip_expert_feedback=args.skip_expert_feedback,
    )


if __name__ == "__main__":
    # Example usage:
    # python mt_he_ours_rebuild.py --generations 15 --k-values 1 3 5 10
    # python mt_he_ours_rebuild.py --generations 10 --skip-expert-feedback  # Faster testing
    main()
