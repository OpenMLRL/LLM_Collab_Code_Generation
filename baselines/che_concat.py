import argparse
import json
import re
import signal
import time
 

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.code_utils import (
    TimeoutException,
    check_aux_call_without_assignment,
    check_aux_function_usage,
    check_function_definition,
    check_syntax,
    cleanup_code,
    concatenate_functions,
    extract_imports_from_prompt,
    extract_specific_function,
    extract_test_cases,
    is_wrapper_function,
    timeout_handler,
)


class QwenCoopHumanEvalDualModelEvaluator:
    def __init__(self, aux_model_name, main_model_name, device="auto"):
        """
        Initialize the dual model evaluator for CoopHumanEval.

        Args:
            aux_model_name: HuggingFace model name for auxiliary function generation
            main_model_name: HuggingFace model name for main function generation
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.aux_model_name = aux_model_name
        self.main_model_name = main_model_name
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

    def aux_function_formatter(self, prompt: str, entry_point: str) -> str:
        """
        Formatter for the auxiliary function generator.
        """
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

def aux(...):
    # your function code here
    return result
"""
        return prompt_text

    def main_function_formatter(self, prompt: str, entry_point: str) -> str:
        """
        Formatter for the main function generator.
        """
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
- You call aux() to assign value to a variable within your function

Your output should follow this format:

def {entry_point}({params_str}):
    # your function code here
    return result
"""
        return prompt_text

    def generate_response(
        self, model, tokenizer, prompt: str, max_new_tokens: int = 256
    ) -> tuple:
        """
        Generate response for a given prompt using specified model and tokenizer.

        Returns:
            tuple: (response, input_tokens, output_tokens)
        """
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
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_tokens = len(generated_tokens)
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip(), input_tokens, output_tokens

    def calculate_new_metrics_per_sample(self, sample_metrics: list, k: int) -> dict:
        """
        🔥 NEW METHOD: Calculate metrics (PassOrNot@k, Accuracy@k, MaxCoop@k) for a single sample.

        Args:
            sample_metrics: List of metrics for all generations of this sample
            k: Number of top generations to consider

        Returns:
            dict: Contains passornot_at_k, accuracy_at_k, maxcoop_at_k
        """
        if not sample_metrics or k <= 0:
            return {"passornot_at_k": 0.0, "accuracy_at_k": 0.0, "maxcoop_at_k": 0.0}

        # Take first k generations (or all if k > number of generations)
        k_metrics = sample_metrics[: min(k, len(sample_metrics))]

        # 🔥 1. PassOrNot@k: Check if any of the k solutions passed ALL tests (FIXED LOGIC)
        passornot_at_k = 0.0
        for metric in k_metrics:
            # 🔥 CRITICAL FIX: A solution passes all tests if passed_tests == total_tests and total_tests > 0
            if (
                metric["total_tests"] > 0
                and metric["passed_tests"] == metric["total_tests"]
            ):
                passornot_at_k = 1.0
                break

        # 🔥 2. Accuracy@k: Find the solution with highest test pass rate among k solutions
        accuracy_at_k = 0.0
        for metric in k_metrics:
            if metric["total_tests"] > 0:
                current_accuracy = metric["passed_tests"] / metric["total_tests"]
                accuracy_at_k = max(accuracy_at_k, current_accuracy)

        # 🔥 3. MaxCoop@k: Find the solution with highest cooperation score among k solutions
        maxcoop_at_k = 0.0
        for metric in k_metrics:
            maxcoop_at_k = max(maxcoop_at_k, metric["bonus_reward"])

        return {
            "passornot_at_k": passornot_at_k,
            "accuracy_at_k": accuracy_at_k,
            "maxcoop_at_k": maxcoop_at_k,
        }

    def evaluate_dual_completion(
        self,
        aux_completion: str,
        main_completion: str,
        test_code: str,
        entry_point: str,
        prompt: str = "",
        sample_id: int = 0,
    ) -> dict:
        """
        Evaluate a dual model completion using the coophumaneval reward logger approach.
        """
        metrics = {
            "sample_id": sample_id,
            "entry_point": entry_point,
            # Level 1 metrics
            "level_1_reward": 0.0,
            "aux_defined": False,
            "main_defined": False,
            # Level 2 metrics
            "level_2_reward": 0.0,
            "syntax_correct": False,
            # Level 3 metrics
            "level_3_reward": 0.0,
            "test_reward": 0.0,
            "passed_tests": 0,
            "total_tests": 0,
            "passed_rate": 0.0,
            "timeout_num": 0,
            "bonus_reward": 0.0,
            "aux_usage_bonus": 0.0,
            "anti_wrapper_bonus": 0.0,
            "called_wo_used_deduction": 0.0,
            # Overall metrics
            "total_reward": 0.0,
            "gated_total_reward": 0.0,
            "execution_error": False,
            # 🔥 NEW: Solution correctness tracking
            "is_correct": False,  # True if passed ALL tests, False otherwise
            "test_results": [],  # List of True/False for each individual test
        }

        TEST_TIMEOUT = 10
        MAX_TIMEOUTS = 3

        # Extract imports from prompt
        imports = extract_imports_from_prompt(prompt)

        # Clean completions
        aux_clean = cleanup_code(aux_completion)
        main_clean = cleanup_code(main_completion)

        # Extract specific functions
        aux_func = extract_specific_function(aux_clean, "aux")
        main_func = extract_specific_function(main_clean, entry_point)

        # ================================================================
        # LEVEL 1: FUNCTION DEFINITION REQUIREMENTS
        # ================================================================

        # Check aux function (+0.4)
        aux_check_passed, _ = check_function_definition(
            aux_clean, "aux", "Aux function"
        )
        if aux_check_passed:
            metrics["level_1_reward"] += 0.4
            metrics["aux_defined"] = True

        # Check main function (+0.6)
        main_check_passed, _ = check_function_definition(
            main_clean, entry_point, f"Main function ({entry_point})"
        )
        if main_check_passed:
            metrics["level_1_reward"] += 0.6
            metrics["main_defined"] = True

        # If main function not defined, stop here
        if not main_check_passed:
            metrics["total_reward"] = metrics["level_1_reward"]
            metrics["gated_total_reward"] = metrics["level_1_reward"]
            return metrics

        # ================================================================
        # LEVEL 2: SYNTAX REQUIREMENTS
        # ================================================================

        # Concatenate functions with imports
        combined_code = concatenate_functions(aux_func, main_func, imports)

        # Check syntax (+0.5)
        syntax_passed, _ = check_syntax(combined_code, "Combined code")
        if syntax_passed:
            metrics["level_2_reward"] = 0.5
            metrics["syntax_correct"] = True

        # If syntax failed, stop here
        if not syntax_passed:
            metrics["total_reward"] = (
                metrics["level_1_reward"] + metrics["level_2_reward"]
            )
            metrics["gated_total_reward"] = metrics["total_reward"]
            return metrics

        # ================================================================
        # LEVEL 3: TEST EXECUTION REQUIREMENTS
        # ================================================================

        # Extract test cases
        test_cases_list = extract_test_cases(test_code, entry_point)
        if not test_cases_list:
            metrics["total_reward"] = (
                metrics["level_1_reward"] + metrics["level_2_reward"]
            )
            metrics["gated_total_reward"] = metrics["total_reward"]
            return metrics

        metrics["total_tests"] = len(test_cases_list)

        # Execute tests
        timeout_count = 0
        passed_tests = 0
        test_results = []  # 🔥 NEW: Track individual test results

        try:
            # Load code definitions
            exec_globals = {}
            exec(combined_code, exec_globals)

            # Verify function is in globals
            if entry_point not in exec_globals:
                metrics["execution_error"] = True
                metrics["total_reward"] = (
                    metrics["level_1_reward"] + metrics["level_2_reward"]
                )
                metrics["gated_total_reward"] = metrics["total_reward"]
                # 🔥 NEW: Fill test_results with False for all tests
                metrics["test_results"] = [False] * len(test_cases_list)
                metrics["is_correct"] = False
                return metrics

            # Run individual test cases
            for test_i, test_case in enumerate(test_cases_list):
                # Check timeout limit
                if timeout_count >= MAX_TIMEOUTS:
                    # 🔥 NEW: Mark remaining tests as False due to timeout
                    remaining_tests = len(test_cases_list) - len(test_results)
                    test_results.extend([False] * remaining_tests)
                    break

                try:
                    # Set timeout for each test
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TEST_TIMEOUT)

                    # Execute test
                    exec(test_case, exec_globals)
                    passed_tests += 1
                    test_results.append(True)  # 🔥 NEW: Record test passed

                    # Clear timeout
                    signal.alarm(0)

                except TimeoutException:
                    signal.alarm(0)
                    timeout_count += 1
                    test_results.append(False)  # 🔥 NEW: Record test failed (timeout)

                except (AssertionError, Exception):
                    signal.alarm(0)
                    test_results.append(False)  # 🔥 NEW: Record test failed
                    # Test failed, continue to next test
                    continue

            # Calculate test metrics
            metrics["passed_tests"] = passed_tests
            metrics["timeout_num"] = timeout_count
            metrics["test_results"] = (
                test_results  # 🔥 NEW: Store individual test results
            )

            if metrics["total_tests"] > 0:
                metrics["passed_rate"] = passed_tests / metrics["total_tests"]
                metrics["test_reward"] = metrics["passed_rate"] * 1.0
                # 🔥 NEW: Solution is correct if ALL tests passed
                metrics["is_correct"] = passed_tests == metrics["total_tests"]

        except Exception:
            # Code loading failed
            signal.alarm(0)
            metrics["execution_error"] = True
            # 🔥 NEW: Fill test_results with False for all tests
            metrics["test_results"] = [False] * len(test_cases_list)
            metrics["is_correct"] = False

        # Level 3 reward is just test reward at this point
        metrics["level_3_reward"] = metrics["test_reward"]

        # ================================================================
        # LEVEL 3 BONUS: COLLABORATION AND COMPLEXITY CHECKS
        # ================================================================

        # Only award bonuses if we have aux function and passed tests
        if passed_tests > 0 and aux_func:
            # Check if main uses aux
            main_uses_aux = check_aux_function_usage(main_func, "aux")

            if main_uses_aux:
                metrics["aux_usage_bonus"] = 0.5
                metrics["bonus_reward"] += 0.5

                # Check if it's NOT a wrapper (anti-wrapper bonus)
                is_wrapper = is_wrapper_function(main_func, "aux")

                if not is_wrapper:
                    metrics["anti_wrapper_bonus"] = 1.0
                    metrics["bonus_reward"] += 1.0

                # Check for aux calls without assignment (deduction)
                has_ignored_calls, ignored_calls = check_aux_call_without_assignment(
                    main_func, "aux"
                )

                if has_ignored_calls:
                    metrics["called_wo_used_deduction"] = 0.5
                    metrics["bonus_reward"] -= 0.5  # Apply deduction to bonus_reward

        # Update level 3 reward to include bonuses (after deduction)
        metrics["level_3_reward"] += metrics["bonus_reward"]

        # ================================================================
        # FINAL REWARD CALCULATION
        # ================================================================

        metrics["total_reward"] = (
            metrics["level_1_reward"]
            + metrics["level_2_reward"]
            + metrics["level_3_reward"]
        )

        # Gated reward considers early stopping
        metrics["gated_total_reward"] = metrics["total_reward"]

        return metrics

    def evaluate_coophumaneval_dual_model(
        self,
        num_samples: int = 31,
        num_generations: int = 1,
        k_values: list = [1, 5, 10],
        save_results: bool = True,
    ):
        """
        Evaluate CoopHumanEval dual model system on the last num_samples.
        """
        print(f"Loading CoopHumanEval dataset...")
        dataset_name = "LovelyBuggies/CoopHumanEval"
        test_data = load_dataset(dataset_name, split="test[66:]")

        # Take the last num_samples
        total_samples = len(test_data)
        start_idx = max(0, total_samples - num_samples)
        test_samples = test_data.select(range(start_idx, total_samples))

        print(
            f"Evaluating on samples {start_idx} to {total_samples - 1} ({len(test_samples)} samples)..."
        )
        print(f"Generating {num_generations} completions per sample...")

        # Store results per sample
        sample_results = []
        all_response_times = []
        all_input_tokens = []
        all_output_tokens = []
        failed_generations = 0

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
                "aux_completions": [],
                "main_completions": [],
                "aux_extracted_functions": [],
                "main_extracted_functions": [],
                "metrics": [],
                "generation_times": [],
                "input_tokens": [],
                "output_tokens": [],
            }

            # Generate multiple completions for this sample
            for gen_idx in range(num_generations):
                try:
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

                    # Extract functions
                    aux_func = extract_specific_function(
                        cleanup_code(aux_response), "aux"
                    )
                    main_func = extract_specific_function(
                        cleanup_code(main_response), entry_point
                    )

                    # Evaluate this completion
                    metrics = self.evaluate_dual_completion(
                        aux_response,
                        main_response,
                        test_code,
                        entry_point,
                        prompt,
                        sample_id=f"{sample_idx}_{gen_idx}",
                    )

                    # Store results
                    sample_data["aux_completions"].append(aux_response)
                    sample_data["main_completions"].append(main_response)
                    sample_data["aux_extracted_functions"].append(aux_func)
                    sample_data["main_extracted_functions"].append(main_func)
                    sample_data["metrics"].append(metrics)
                    sample_data["generation_times"].append(
                        (agent1_time + agent2_time) / 2
                    )
                    sample_data["input_tokens"].append(
                        aux_input_tokens + main_input_tokens
                    )
                    sample_data["output_tokens"].append(
                        aux_output_tokens + main_output_tokens
                    )

                    # Add to global lists
                    all_response_times.append((agent1_time + agent2_time) / 2)
                    all_input_tokens.append(aux_input_tokens + main_input_tokens)
                    all_output_tokens.append(aux_output_tokens + main_output_tokens)

                except Exception as e:
                    print(f"Failed on sample {sample_idx}, generation {gen_idx}: {e}")
                    failed_generations += 1

                    # Add empty results
                    empty_metrics = {
                        "sample_id": f"{sample_idx}_{gen_idx}",
                        "entry_point": entry_point,
                        "level_1_reward": 0.0,
                        "aux_defined": False,
                        "main_defined": False,
                        "level_2_reward": 0.0,
                        "syntax_correct": False,
                        "level_3_reward": 0.0,
                        "test_reward": 0.0,
                        "passed_tests": 0,
                        "total_tests": 0,
                        "passed_rate": 0.0,
                        "timeout_num": 0,
                        "bonus_reward": 0.0,
                        "aux_usage_bonus": 0.0,
                        "anti_wrapper_bonus": 0.0,
                        "called_wo_used_deduction": 0.0,
                        "total_reward": 0.0,
                        "gated_total_reward": 0.0,
                        "execution_error": True,
                        # 🔥 NEW: Add correctness tracking for failed generations
                        "is_correct": False,
                        "test_results": [],
                    }

                    sample_data["aux_completions"].append("")
                    sample_data["main_completions"].append("")
                    sample_data["aux_extracted_functions"].append("")
                    sample_data["main_extracted_functions"].append("")
                    sample_data["metrics"].append(empty_metrics)
                    sample_data["generation_times"].append(0.0)
                    sample_data["input_tokens"].append(0)
                    sample_data["output_tokens"].append(0)

            sample_results.append(sample_data)

        print(f"Generation complete. Failed generations: {failed_generations}")

        # Calculate aggregated metrics
        print("Calculating metrics...")

        # Collect all metrics for overall statistics
        all_metrics = []
        for sample_data in sample_results:
            all_metrics.extend(sample_data["metrics"])

        # 🔥 COMPLETELY REPLACED: Calculate NEW metrics instead of old incorrect pass@k
        # Calculate NEW metrics: PassOrNot@k, Accuracy@k, MaxCoop@k
        passornot_at_k = {}  # 🔥 NEW: PassOrNot@k (all tests passed)
        accuracy_at_k = {}  # 🔥 NEW: Accuracy@k (best accuracy)
        maxcoop_at_k = {}  # 🔥 NEW: MaxCoop@k (best cooperation)

        for k in k_values:
            if k <= num_generations:
                # 🔥 NEW: Initialize lists for new metrics
                passornot_results = []  # 🔥 NEW
                accuracy_results = []  # 🔥 NEW
                maxcoop_results = []  # 🔥 NEW

                for sample_data in sample_results:
                    # 🔥 NEW: Calculate NEW metrics for this sample using our new method
                    new_metrics = self.calculate_new_metrics_per_sample(
                        sample_data["metrics"], k
                    )

                    # 🔥 NEW: Add results to new metric lists
                    passornot_results.append(new_metrics["passornot_at_k"])
                    accuracy_results.append(new_metrics["accuracy_at_k"])
                    maxcoop_results.append(new_metrics["maxcoop_at_k"])

                # 🔥 NEW: Average NEW metrics across all samples
                passornot_at_k[f"PassOrNot@{k}"] = np.mean(passornot_results)
                accuracy_at_k[f"Accuracy@{k}"] = np.mean(accuracy_results)
                maxcoop_at_k[f"MaxCoop@{k}"] = np.mean(maxcoop_results)

        # Calculate other metrics
        structure_extraction_rate = np.mean([m["main_defined"] for m in all_metrics])
        syntax_correct_rate = np.mean([m["syntax_correct"] for m in all_metrics])
        cooperation_rate = np.mean([m["bonus_reward"] > 0 for m in all_metrics])

        # 🔥 NEW: Calculate full test pass rate (all tests passed) - FIXED METRIC
        full_test_pass_rate = np.mean(
            [
                (m["total_tests"] > 0 and m["passed_tests"] == m["total_tests"])
                for m in all_metrics
            ]
        )

        # 🔥 NEW: Calculate solution correctness rate
        solution_correctness_rate = np.mean([m["is_correct"] for m in all_metrics])

        # Reward statistics
        avg_level_1_reward = np.mean([m["level_1_reward"] for m in all_metrics])
        avg_level_2_reward = np.mean([m["level_2_reward"] for m in all_metrics])
        avg_level_3_reward = np.mean([m["level_3_reward"] for m in all_metrics])
        avg_total_reward = np.mean([m["total_reward"] for m in all_metrics])
        avg_bonus_reward = np.mean([m["bonus_reward"] for m in all_metrics])

        # Token statistics
        avg_input_tokens = np.mean(all_input_tokens) if all_input_tokens else 0
        avg_output_tokens = np.mean(all_output_tokens) if all_output_tokens else 0
        avg_response_time = np.mean(all_response_times) if all_response_times else 0

        # Compile results
        aggregated_metrics = {
            # Basic metrics
            "aux_model_name": self.aux_model_name,
            "main_model_name": self.main_model_name,
            "num_samples": len(test_samples),
            "num_generations_per_sample": num_generations,
            "total_generations": len(all_metrics),
            "failed_generations": failed_generations,
            "success_rate": (
                (len(all_metrics) - failed_generations) / len(all_metrics)
                if all_metrics
                else 0
            ),
            # Performance metrics
            "structure_extraction_rate": structure_extraction_rate,
            "syntax_correct_rate": syntax_correct_rate,
            "full_test_pass_rate": full_test_pass_rate,
            "solution_correctness_rate": solution_correctness_rate,  # 🔥 NEW: Overall correctness rate
            "cooperation_rate": cooperation_rate,
            # 🔥 REPLACED: Main metrics (removed all old incorrect pass@k metrics)
            "passornot_at_k": passornot_at_k,  # 🔥 PassOrNot@k: at least one solution passes ALL tests
            "accuracy_at_k": accuracy_at_k,  # 🔥 Accuracy@k: best accuracy among k solutions
            "maxcoop_at_k": maxcoop_at_k,  # 🔥 MaxCoop@k: best cooperation score among k solutions
            # Reward metrics
            "avg_level_1_reward": avg_level_1_reward,
            "avg_level_2_reward": avg_level_2_reward,
            "avg_level_3_reward": avg_level_3_reward,
            "avg_total_reward": avg_total_reward,
            "avg_bonus_reward": avg_bonus_reward,
            # Token and timing metrics
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_input_tokens + avg_output_tokens,
            "avg_response_time": avg_response_time,
            "total_time": np.sum(all_response_times),
            # Test execution metrics
            "avg_passed_tests": np.mean([m["passed_tests"] for m in all_metrics]),
            "avg_total_tests": np.mean([m["total_tests"] for m in all_metrics]),
            "avg_timeout_num": np.mean([m["timeout_num"] for m in all_metrics]),
        }

        # Print results
        print("\n" + "=" * 60)
        print("HUMANEVAL DUAL MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Aux Model: {self.aux_model_name}")
        print(f"Main Model: {self.main_model_name}")
        print(
            f"Samples evaluated: {len(test_samples)} (indices {start_idx}-{total_samples - 1})"
        )
        print(f"Generations per sample: {num_generations}")
        print(f"Total generations: {len(all_metrics)}")
        print(f"Success rate: {aggregated_metrics['success_rate']:.1%}")
        print()

        print("TIMING AND TOKEN METRICS:")
        print(f"Average response time: {aggregated_metrics['avg_response_time']:.2f}s")
        print(f"Total time: {aggregated_metrics['total_time']:.1f}s")
        print(f"Average input tokens: {aggregated_metrics['avg_input_tokens']:.1f}")
        print(f"Average output tokens: {aggregated_metrics['avg_output_tokens']:.1f}")
        print(f"Average total tokens: {aggregated_metrics['avg_total_tokens']:.1f}")
        print()

        print("PERFORMANCE METRICS:")
        print(
            f"Structure extraction rate: {aggregated_metrics['structure_extraction_rate']:.3f}"
        )
        print(f"Syntax correct rate: {aggregated_metrics['syntax_correct_rate']:.3f}")
        print(f"Full test pass rate: {aggregated_metrics['full_test_pass_rate']:.3f}")
        print(
            f"Solution correctness rate: {aggregated_metrics['solution_correctness_rate']:.3f}"
        )  # 🔥 NEW
        print(f"Cooperation rate: {aggregated_metrics['cooperation_rate']:.3f}")
        print()

        print(
            "🔥 MAIN EVALUATION METRICS (ADDRESSING THE ORIGINAL PROBLEM):"
        )  # 🔥 HIGHLIGHTED NEW SECTION
        print(
            "PassOrNot@k (at least one solution passes ALL tests):"
        )  # 🔥 CLEAR DEFINITION
        for k, v in passornot_at_k.items():
            print(f"  {k}: {v:.3f}")
        print(
            "Accuracy@k (best test accuracy among k solutions):"
        )  # 🔥 CLEAR DEFINITION
        for k, v in accuracy_at_k.items():
            print(f"  {k}: {v:.3f}")
        print(
            "MaxCoop@k (best cooperation score among k solutions):"
        )  # 🔥 CLEAR DEFINITION
        for k, v in maxcoop_at_k.items():
            print(f"  {k}: {v:.3f}")
        print()

        print("REWARD METRICS:")
        print(f"Average Level 1 reward: {aggregated_metrics['avg_level_1_reward']:.3f}")
        print(f"Average Level 2 reward: {aggregated_metrics['avg_level_2_reward']:.3f}")
        print(f"Average Level 3 reward: {aggregated_metrics['avg_level_3_reward']:.3f}")
        print(f"Average Total reward: {aggregated_metrics['avg_total_reward']:.3f}")
        print(f"Average Bonus reward: {aggregated_metrics['avg_bonus_reward']:.3f}")
        print()

        print("TEST EXECUTION METRICS:")
        print(f"Average passed tests: {aggregated_metrics['avg_passed_tests']:.1f}")
        print(f"Average total tests: {aggregated_metrics['avg_total_tests']:.1f}")
        print(f"Average timeouts: {aggregated_metrics['avg_timeout_num']:.1f}")

        # Save detailed results
        if save_results:
            results = {
                "aggregated_metrics": aggregated_metrics,
                "sample_results": sample_results,
                "evaluation_config": {
                    "num_samples": num_samples,
                    "num_generations": num_generations,
                    "k_values": k_values,
                    "sample_indices": f"{start_idx}-{total_samples - 1}",
                },
            }

            filename = f"coophumaneval_concatenation_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, sample_results


def main():
    parser = argparse.ArgumentParser(description="CoopHumanEval Dual Model Evaluation")
    parser.add_argument(
        "--aux-model", default="Qwen/Qwen2.5-Coder-3B", help="Auxiliary model name"
    )
    parser.add_argument(
        "--main-model", default="Qwen/Qwen2.5-Coder-3B", help="Main model name"
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

    args = parser.parse_args()

    # Initialize dual model evaluator
    evaluator = QwenCoopHumanEvalDualModelEvaluator(
        aux_model_name=args.aux_model,
        main_model_name=args.main_model,
        device=args.device,
    )

    # Run evaluation
    aggregated_metrics, sample_results = evaluator.evaluate_coophumaneval_dual_model(
        num_samples=args.samples,
        num_generations=args.generations,
        k_values=args.k_values,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    # python experiments/evaluation/che_concat.py --generations 15 --k-values 1 3 5 10
    main()
