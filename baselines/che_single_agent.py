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
    check_function_definition,
    check_syntax,
    cleanup_code,
    extract_imports_from_prompt,
    extract_specific_function,
    extract_test_cases,
    timeout_handler,
)


class QwenCoopHumanEvalSingleAgentBaseline:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B", device="auto"):
        """
        Initialize the Qwen model for CoopHumanEval single agent baseline evaluation.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_function_params_from_prompt(self, prompt_text):
        """Extract function parameters from the prompt text."""
        # Match pattern like: def function_name(param1: type1, param2: type2) -> return_type:
        match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
        if match:
            params_str = match.group(1)
            # Clean up parameters
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            return params
        return []

    def create_prompt(self, prompt: str, entry_point: str) -> str:
        """
        Create a more specific prompt for function generation to improve extraction.

        Args:
            prompt: The original CoopHumanEval prompt
            entry_point: The main function name
        """
        # Extract function parameters from the prompt
        params_list = self.extract_function_params_from_prompt(prompt)
        params_str = ", ".join(params_list)

        # Create structured prompt following the specified format
        generation_prompt = f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Implement ONLY the '{entry_point}' function as specified
- Make sure to include proper return statement

Your output should follow this format:

def {entry_point}({params_str}):
    # your function code here
    return result
"""
        return generation_prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> tuple:
        """
        Generate response for a given prompt.

        Returns:
            tuple: (response, input_tokens, output_tokens)
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,  # Slightly lower temperature for more focused code
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer,
            )

        # Extract only the generated code
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_tokens = len(generated_tokens)
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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

        # 🔥 3. MaxCoop@k: For single agent, cooperation score is always 0 (no aux function)
        maxcoop_at_k = 0.0  # Single agent has no cooperation

        return {
            "passornot_at_k": passornot_at_k,
            "accuracy_at_k": accuracy_at_k,
            "maxcoop_at_k": maxcoop_at_k,
        }

    def extract_function_from_response(
        self, response: str, function_name: str
    ) -> tuple:
        """
        Extract function definition from model response using utility functions.

        Args:
            response: The generated response
            function_name: Name of the function to extract

        Returns:
            tuple: (extracted_function_code, extraction_successful)
        """
        if not response or not response.strip():
            return "", False

        # First, try to clean up the response using the utility function
        cleaned_response = cleanup_code(response)

        if cleaned_response:
            # Try to extract the specific function
            extracted_function = extract_specific_function(
                cleaned_response, function_name
            )
            if extracted_function:
                return extracted_function, True

        # If cleanup didn't work, try with original response
        extracted_function = extract_specific_function(response, function_name)
        if extracted_function:
            return extracted_function, True

        # If specific extraction failed, try manual extraction with more permissive approach
        # Look for function definition pattern
        pattern = rf"def\s+{re.escape(function_name)}\s*\([^)]*\):"
        match = re.search(pattern, response)

        if match:
            # Found function definition, extract it
            start_pos = match.start()
            lines = response[start_pos:].split("\n")

            function_lines = []
            indent_level = None

            for i, line in enumerate(lines):
                if i == 0:  # First line (def line)
                    function_lines.append(line)
                    continue

                if line.strip() == "":  # Empty line
                    function_lines.append(line)
                    continue

                # Determine indentation level from first non-empty line after def
                if indent_level is None and line.strip():
                    current_indent = len(line) - len(line.lstrip())
                    def_indent = len(lines[0]) - len(lines[0].lstrip())
                    if current_indent > def_indent:
                        indent_level = current_indent
                    else:
                        # This line is not indented relative to def
                        break

                # Check if this line is still part of the function
                if indent_level is not None:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent < indent_level:
                        # This line is less indented than the function body
                        break

                function_lines.append(line)

            extracted = "\n".join(function_lines).strip()
            if extracted and f"def {function_name}" in extracted:
                return extracted, True

        # Final fallback - return original response if it contains the function name
        if f"def {function_name}" in response:
            return (
                response,
                False,
            )  # Mark as unsuccessful extraction but return something

        return "", False

    def evaluate_single_completion(
        self,
        completion: str,
        test_code: str,
        entry_point: str,
        prompt: str = "",
        sample_id: int = 0,
        extraction_successful: bool = False,
    ) -> dict:
        """
        Evaluate a single completion against test cases.

        Args:
            completion: The generated function code
            test_code: Test cases to run
            entry_point: Function name
            prompt: Original prompt (for import extraction)
            sample_id: Sample identifier
            extraction_successful: Whether function extraction was successful

        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            "sample_id": sample_id,
            "entry_point": entry_point,
            "extraction_successful": extraction_successful,
            "syntax_correct": False,
            "test_results": [],  # List of boolean results for each test
            "passed_tests": 0,
            "total_tests": 0,
            "timeout_num": 0,
            "execution_error": False,
            # 🔥 NEW: Solution correctness tracking
            "is_correct": False,  # True if passed ALL tests, False otherwise
        }

        TEST_TIMEOUT = 10
        MAX_TIMEOUTS = 3

        if not completion or not completion.strip():
            return metrics

        # Extract imports from prompt
        imports = extract_imports_from_prompt(prompt)

        # Use the completion as-is since it's already been processed by extract_function_from_response
        completion_clean = completion

        # Check function definition using utility
        func_defined, func_def_msg = check_function_definition(
            completion_clean, entry_point, f"Function ({entry_point})"
        )

        if not func_defined:
            return metrics

        # Combine with imports
        combined_code = (
            imports + "\n" + completion_clean if imports else completion_clean
        )

        # Check syntax using utility
        syntax_passed, syntax_msg = check_syntax(combined_code, "Function code")

        if syntax_passed:
            metrics["syntax_correct"] = True
        else:
            # Try without imports in case imports are causing issues
            syntax_passed, _ = check_syntax(completion_clean, "Function code")
            if syntax_passed:
                metrics["syntax_correct"] = True
                combined_code = completion_clean
            else:
                return metrics

        # Extract test cases
        test_cases_list = extract_test_cases(test_code, entry_point)

        if not test_cases_list:
            return metrics

        metrics["total_tests"] = len(test_cases_list)

        # Execute tests
        timeout_count = 0
        passed_tests = 0
        test_results = []  # 🔥 ALREADY EXISTS: Track individual test results

        try:
            # Load code definitions
            exec_globals = {}
            exec(combined_code, exec_globals)

            # Verify function is in globals
            if entry_point not in exec_globals:
                metrics["execution_error"] = True
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
                    test_results.append(True)  # 🔥 ALREADY EXISTS: Record test passed

                    # Clear timeout
                    signal.alarm(0)

                except TimeoutException:
                    signal.alarm(0)
                    timeout_count += 1
                    test_results.append(
                        False
                    )  # 🔥 ALREADY EXISTS: Record test failed (timeout)

                except (AssertionError, Exception):
                    signal.alarm(0)
                    test_results.append(False)  # 🔥 ALREADY EXISTS: Record test failed

            # Update metrics
            metrics["passed_tests"] = passed_tests
            metrics["timeout_num"] = timeout_count
            metrics["test_results"] = test_results

            # 🔥 NEW: Solution is correct if ALL tests passed
            if metrics["total_tests"] > 0:
                metrics["is_correct"] = passed_tests == metrics["total_tests"]

        except Exception:
            # Code loading failed
            signal.alarm(0)
            metrics["execution_error"] = True
            # 🔥 NEW: Fill test_results with False for all tests
            metrics["test_results"] = [False] * len(test_cases_list)
            metrics["is_correct"] = False

        return metrics

    def evaluate_coophumaneval_baseline(
        self,
        num_samples: int = 31,
        num_generations: int = 1,
        k_values: list = [1, 5, 10],
        save_results: bool = True,
    ):
        """
        Evaluate CoopHumanEval single agent baseline on the last num_samples.

        Args:
            num_samples: Number of test samples to evaluate (default 31)
            num_generations: Number of completions to generate per sample (default 1)
            k_values: List of k values for pass@k calculation (default [1, 5, 10])
            save_results: Whether to save detailed results to JSON
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
                "completions": [],
                "extracted_functions": [],
                "extraction_success": [],
                "metrics": [],
                "generation_times": [],
                "input_tokens": [],
                "output_tokens": [],
            }

            # Generate multiple completions for this sample
            for gen_idx in range(num_generations):
                try:
                    # Generate function
                    generation_prompt = self.create_prompt(prompt, entry_point)
                    start_time = time.time()
                    response, input_tokens, output_tokens = self.generate_response(
                        generation_prompt
                    )
                    end_time = time.time()

                    # Extract function definition using improved method
                    function_code, extraction_successful = (
                        self.extract_function_from_response(response, entry_point)
                    )

                    # Evaluate this completion
                    metrics = self.evaluate_single_completion(
                        function_code,
                        test_code,
                        entry_point,
                        prompt,
                        sample_id=f"{sample_idx}_{gen_idx}",
                        extraction_successful=extraction_successful,
                    )

                    # Store results
                    sample_data["completions"].append(response)
                    sample_data["extracted_functions"].append(function_code)
                    sample_data["extraction_success"].append(extraction_successful)
                    sample_data["metrics"].append(metrics)
                    sample_data["generation_times"].append(end_time - start_time)
                    sample_data["input_tokens"].append(input_tokens)
                    sample_data["output_tokens"].append(output_tokens)

                    # Add to global lists
                    all_response_times.append(end_time - start_time)
                    all_input_tokens.append(input_tokens)
                    all_output_tokens.append(output_tokens)

                except Exception as e:
                    print(f"Failed on sample {sample_idx}, generation {gen_idx}: {e}")
                    failed_generations += 1

                    # Add empty results
                    sample_data["completions"].append("")
                    sample_data["extracted_functions"].append("")
                    sample_data["extraction_success"].append(False)
                    sample_data["metrics"].append(
                        {
                            "sample_id": f"{sample_idx}_{gen_idx}",
                            "entry_point": entry_point,
                            "extraction_successful": False,
                            "syntax_correct": False,
                            "test_results": [],
                            "passed_tests": 0,
                            "total_tests": 0,
                            "timeout_num": 0,
                            "execution_error": True,
                            # 🔥 NEW: Add correctness tracking for failed generations
                            "is_correct": False,
                        }
                    )
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
        maxcoop_at_k = {}  # 🔥 NEW: MaxCoop@k (always 0 for single agent)

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
        structure_extraction_rate = np.mean(
            [m["extraction_successful"] for m in all_metrics]
        )
        syntax_correct_rate = np.mean([m["syntax_correct"] for m in all_metrics])

        # 🔥 NEW: Calculate full test pass rate (all tests passed) - FIXED METRIC
        full_test_pass_rate = np.mean(
            [
                (m["total_tests"] > 0 and m["passed_tests"] == m["total_tests"])
                for m in all_metrics
            ]
        )

        # 🔥 NEW: Calculate solution correctness rate
        solution_correctness_rate = np.mean([m["is_correct"] for m in all_metrics])

        # Token statistics
        avg_input_tokens = np.mean(all_input_tokens) if all_input_tokens else 0
        avg_output_tokens = np.mean(all_output_tokens) if all_output_tokens else 0
        avg_response_time = np.mean(all_response_times) if all_response_times else 0

        # Calculate function length statistics
        all_extracted_functions = []
        for sample_data in sample_results:
            all_extracted_functions.extend(sample_data["extracted_functions"])

        func_lengths = [len(func.split()) for func in all_extracted_functions if func]
        avg_function_length = np.mean(func_lengths) if func_lengths else 0

        # Compile results
        aggregated_metrics = {
            # Basic metrics
            "model_name": self.model_name,
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
            # 🔥 REPLACED: Main metrics (removed all old incorrect pass@k metrics)
            "passornot_at_k": passornot_at_k,  # 🔥 PassOrNot@k: at least one solution passes ALL tests
            "accuracy_at_k": accuracy_at_k,  # 🔥 Accuracy@k: best accuracy among k solutions
            "maxcoop_at_k": maxcoop_at_k,  # 🔥 MaxCoop@k: always 0 for single agent (no cooperation)
            # Token and timing metrics
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_input_tokens + avg_output_tokens,
            "avg_response_time": avg_response_time,
            "total_time": np.sum(all_response_times),
            "avg_function_length": avg_function_length,
            # Test execution metrics
            "avg_passed_tests": np.mean([m["passed_tests"] for m in all_metrics]),
            "avg_total_tests": np.mean([m["total_tests"] for m in all_metrics]),
            "avg_timeout_num": np.mean([m["timeout_num"] for m in all_metrics]),
        }

        # Print results
        print("\n" + "=" * 60)
        print("HUMANEVAL SINGLE AGENT BASELINE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {self.model_name}")
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
        print(
            f"Average function length: {aggregated_metrics['avg_function_length']:.1f} words"
        )
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
            "MaxCoop@k (best cooperation score - always 0 for single agent):"
        )  # 🔥 CLEAR DEFINITION
        for k, v in maxcoop_at_k.items():
            print(f"  {k}: {v:.3f}")
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

            filename = f"coophumaneval_baseline_results_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {filename}")

        return aggregated_metrics, sample_results


def main():
    # --------------------------------------------------------------
    # CLI: parse arguments
    # --------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="CoopHumanEval Single Agent Baseline Evaluation"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B", help="Model name")
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

    # --------------------------------------------------------------
    # Initialize evaluator
    # --------------------------------------------------------------
    evaluator = QwenCoopHumanEvalSingleAgentBaseline(
        model_name=args.model, device=args.device
    )

    # --------------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------------
    aggregated_metrics, sample_results = evaluator.evaluate_coophumaneval_baseline(
        num_samples=args.samples,
        num_generations=args.generations,
        k_values=args.k_values,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    # python experiments/baselines/che_single_agent.py --generations 15 --k-values 1 3 5 10
    main()
