import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
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
)

# Reuse training-consistent first-turn prompts for aux/main
from LLM_Collab_Code_Generation.train_magrpo import (
    aux_function_formatter as aux_formatter,
    main_function_formatter as main_formatter,
)


def load_model_and_tokenizer(model_name: str, device_preference: str = "auto"):
    device = (
        device_preference
        if device_preference != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device: str, prompt: str, max_new_tokens: int = 256,
             temperature: float = 0.8, top_p: float = 0.95) -> Tuple[str, int, int, float]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return response.strip(), int(inputs["input_ids"].shape[1]), int(len(gen_tokens)), float(dt)


def evaluate_dual_completion(
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
    prompt: str = "",
    per_test_timeout: int = 10,
) -> Dict[str, Any]:
    """Evaluate aux+main completions using the same checks as rewards.code_rewards.

    Returns detailed metrics including level rewards and pass/fail counts.
    """
    metrics = {
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
        "execution_error": False,
        "is_correct": False,
        "test_results": [],
    }

    imports = extract_imports_from_prompt(prompt)
    aux_clean = cleanup_code(aux_completion)
    main_clean = cleanup_code(main_completion)
    aux_func = extract_specific_function(aux_clean, "aux")
    main_func = extract_specific_function(main_clean, entry_point)

    # Level 1
    aux_ok, _ = check_function_definition(aux_clean, "aux", "Aux function")
    if aux_ok:
        metrics["level_1_reward"] += 0.4
        metrics["aux_defined"] = True
    main_ok, _ = check_function_definition(main_clean, entry_point, f"Main function ({entry_point})")
    if main_ok:
        metrics["level_1_reward"] += 0.6
        metrics["main_defined"] = True
    if not main_ok:
        metrics["total_reward"] = metrics["level_1_reward"]
        metrics["gated_total_reward"] = metrics["level_1_reward"]
        return metrics

    # Level 2
    combined_code = concatenate_functions(aux_func, main_func, imports)
    syntax_ok, _ = check_syntax(combined_code, "Combined code")
    if syntax_ok:
        metrics["level_2_reward"] = 0.5
        metrics["syntax_correct"] = True
    else:
        metrics["total_reward"] = metrics["level_1_reward"] + metrics["level_2_reward"]
        metrics["gated_total_reward"] = metrics["total_reward"]
        return metrics

    # Level 3: tests
    test_cases_list = extract_test_cases(test_code, entry_point)
    if not test_cases_list:
        metrics["total_reward"] = metrics["level_1_reward"] + metrics["level_2_reward"]
        metrics["gated_total_reward"] = metrics["total_reward"]
        return metrics

    metrics["total_tests"] = len(test_cases_list)
    timeout_count = 0
    passed_tests = 0
    results: List[bool] = []
    try:
        exec_globals = {}
        exec(combined_code, exec_globals)
        if entry_point not in exec_globals:
            metrics["execution_error"] = True
            metrics["total_reward"] = metrics["level_1_reward"] + metrics["level_2_reward"]
            metrics["gated_total_reward"] = metrics["total_reward"]
            metrics["test_results"] = [False] * len(test_cases_list)
            return metrics

        import signal

        def _handler(signum, frame):
            raise TimeoutException("timeout")

        for test_case in test_cases_list:
            if timeout_count >= 3:
                results.extend([False] * (len(test_cases_list) - len(results)))
                break
            try:
                signal.signal(signal.SIGALRM, _handler)
                signal.alarm(per_test_timeout)
                exec(test_case, exec_globals)
                signal.alarm(0)
                passed_tests += 1
                results.append(True)
            except TimeoutException:
                signal.alarm(0)
                timeout_count += 1
                results.append(False)
            except Exception:
                signal.alarm(0)
                results.append(False)
    except Exception:
        metrics["execution_error"] = True
        results = [False] * len(test_cases_list)

    metrics["passed_tests"] = passed_tests
    metrics["timeout_num"] = timeout_count
    metrics["test_results"] = results
    if metrics["total_tests"] > 0:
        metrics["passed_rate"] = passed_tests / metrics["total_tests"]
        metrics["test_reward"] = metrics["passed_rate"] * 1.0
        metrics["is_correct"] = (passed_tests == metrics["total_tests"])

    # Bonuses
    if passed_tests > 0 and aux_func:
        if check_aux_function_usage(main_func, "aux"):
            metrics["aux_usage_bonus"] = 0.5
            metrics["bonus_reward"] += 0.5
            if not is_wrapper_function(main_func, "aux"):
                metrics["anti_wrapper_bonus"] = 1.0
                metrics["bonus_reward"] += 1.0
            has_ignored_calls, _ = check_aux_call_without_assignment(main_func, "aux")
            if has_ignored_calls:
                metrics["called_wo_used_deduction"] = 0.5
                metrics["bonus_reward"] -= 0.5

    metrics["level_3_reward"] = metrics["test_reward"] + metrics["bonus_reward"]
    metrics["total_reward"] = (
        metrics["level_1_reward"] + metrics["level_2_reward"] + metrics["level_3_reward"]
    )
    metrics["gated_total_reward"] = metrics["total_reward"]
    return metrics


def compute_pass_at_k(sample_metrics: List[List[Dict[str, Any]]], ks: List[int]) -> Dict[str, float]:
    results = {}
    if not sample_metrics:
        for k in ks:
            results[f"pass@{k}"] = 0.0
        return results
    for k in ks:
        cnt = 0
        for gens in sample_metrics:
            kk = min(k, len(gens))
            ok = any(m.get("is_correct", False) for m in gens[:kk])
            cnt += 1 if ok else 0
        results[f"pass@{k}"] = cnt / len(sample_metrics)
    return results


def append_result_jsonl(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def sequential_main_prompt(prompt: str, entry_point: str, aux_code: str) -> str:
    """Main prompt for sequential pipeline where main sees aux implementation.

    Stays close to the training formatter while surfacing the aux code context.
    """
    # Infer parameter list for nice formatting
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", prompt or "")
    params_str = match.group(1) if match else ""
    return (
        f"Solve this coding problem by implementing the required function.\n\n"
        f"Problem:\n{prompt}\n\n"
        f"You have access to this helper function that has already been implemented:\n\n{aux_code}\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"- Output ONLY the function code, no explanations or examples\n"
        f"- Do NOT include markdown code blocks (```python)\n"
        f"- Do NOT include any text before or after the function\n"
        f"- Do NOT include test cases or example usage\n"
        f"- Do NOT redefine the aux() function (it's provided above)\n"
        f"- Implement ONLY the '{entry_point}' function as specified\n"
        f"- You can call aux() to get useful data for solving the problem\n\n"
        f"Your output should follow this format:\n\n"
        f"def {entry_point}({params_str}):\n    # your function code here\n    return result\n"
    )


def discussion_round2_prompts(prompt: str, entry_point: str, aux_round1: str, main_round1: str) -> Tuple[str, str]:
    """Construct round-2 prompts mirroring the existing discussion baseline style."""
    # Aux prompt (improve aux based on main usage)
    aux_p = (
        f"Improve your helper function based on how it's being used by the main function.\n\n"
        f"Problem:\n{prompt}\n\n"
        f"The main function implementation from another agent:\n{main_round1}\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"- Output ONLY the improved aux function code, no explanations\n"
        f"- Do NOT include markdown code blocks (```python)\n"
        f"- Do NOT include any text before or after the function\n"
        f"- Create an improved helper function named 'aux' that better assists the main function\n\n"
        f"Your output should follow this format:\n\n"
        f"def aux(...):\n    # your improved function code here\n    return result\n"
    )

    # Main prompt (improve main based on aux implementation)
    # Infer parameter list for nice formatting
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", prompt or "")
    params_str = match.group(1) if match else ""
    main_p = (
        f"Improve your main function implementation based on the helper function provided.\n\n"
        f"Problem:\n{prompt}\n\n"
        f"The auxiliary function implementation from another agent:\n{aux_round1}\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"- Output ONLY the improved main function code, no explanations\n"
        f"- Do NOT include markdown code blocks (```python)\n"
        f"- Do NOT include any text before or after the function\n"
        f"- Do NOT redefine the aux() function (it's provided above)\n"
        f"- Implement ONLY the improved '{entry_point}' function\n\n"
        f"Your output should follow this format:\n\n"
        f"def {entry_point}({params_str}):\n    # your improved function code here\n    return result\n"
    )

    return aux_p, main_p


def main():
    parser = argparse.ArgumentParser(description="Multi-agent TTI baselines")
    parser.add_argument("--dataset", required=True, help="HF dataset id (HE/CHE)")
    parser.add_argument("--mode", required=True, choices=["naive_concat", "sequential_pipeline", "one_round_discussion"], help="TTI mode")
    parser.add_argument("--aux-model", required=True, help="Auxiliary model name (HF id)")
    parser.add_argument("--main-model", required=True, help="Main model name (HF id)")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--generations", type=int, default=3, help="Generations per sample")
    parser.add_argument("--result-json", required=True, help="Output JSONL path to append a summary line")
    args = parser.parse_args()

    dataset_name = args.dataset
    mode = args.mode

    # Data
    test_data = load_dataset(dataset_name, split="test")
    total = len(test_data)
    start_idx = max(0, total - args.samples)
    samples = test_data.select(range(start_idx, total))

    # Models
    aux_model, aux_tok, aux_dev = load_model_and_tokenizer(args.aux_model)
    main_model, main_tok, main_dev = load_model_and_tokenizer(args.main_model)

    # Storage
    per_sample_gen_metrics: List[List[Dict[str, Any]]] = []
    avg_times: List[float] = []
    total_output_tokens = 0

    for item in samples:
        prompt = item["prompt"]
        entry_point = item.get("entry_point", "")
        test_code = item.get("test", "")

        sample_metrics: List[Dict[str, Any]] = []

        for _ in range(args.generations):
            if mode == "naive_concat":
                # Turn 1 only: independent aux/main
                aux_prompt = aux_formatter({"prompt": prompt, "entry_point": entry_point})
                aux_resp, a_in, a_out, a_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt)
                main_prompt = main_formatter({"prompt": prompt, "entry_point": entry_point})
                main_resp, m_in, m_out, m_dt = generate(main_model, main_tok, main_dev, main_prompt)

                total_output_tokens += (a_out + m_out)
                avg_times.append((a_dt + m_dt) / 2)

                metrics = evaluate_dual_completion(
                    aux_resp, main_resp, test_code, entry_point, prompt
                )
                sample_metrics.append(metrics)

            elif mode == "sequential_pipeline":
                # Turn 1 only: main sees aux implementation
                aux_prompt = aux_formatter({"prompt": prompt, "entry_point": entry_point})
                aux_resp, a_in, a_out, a_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt)
                aux_func = extract_specific_function(cleanup_code(aux_resp), "aux") or aux_resp

                main_prompt = sequential_main_prompt(prompt, entry_point, aux_func)
                main_resp, m_in, m_out, m_dt = generate(main_model, main_tok, main_dev, main_prompt)

                total_output_tokens += (a_out + m_out)
                avg_times.append(a_dt + m_dt)

                metrics = evaluate_dual_completion(
                    aux_resp, main_resp, test_code, entry_point, prompt
                )
                sample_metrics.append(metrics)

            else:  # one_round_discussion
                # Round 1: independent
                aux_prompt_r1 = aux_formatter({"prompt": prompt, "entry_point": entry_point})
                aux_r1, a1_in, a1_out, a1_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt_r1)
                main_prompt_r1 = main_formatter({"prompt": prompt, "entry_point": entry_point})
                main_r1, m1_in, m1_out, m1_dt = generate(main_model, main_tok, main_dev, main_prompt_r1)

                aux_func_r1 = extract_specific_function(cleanup_code(aux_r1), "aux") or aux_r1
                main_func_r1 = extract_specific_function(cleanup_code(main_r1), entry_point) or main_r1

                # Round 2: cross-referenced improvement
                aux_prompt_r2, main_prompt_r2 = discussion_round2_prompts(
                    prompt, entry_point, aux_func_r1, main_func_r1
                )
                aux_r2, a2_in, a2_out, a2_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt_r2)
                main_r2, m2_in, m2_out, m2_dt = generate(main_model, main_tok, main_dev, main_prompt_r2)

                total_output_tokens += (a1_out + m1_out + a2_out + m2_out)
                avg_times.append(a1_dt + m1_dt + a2_dt + m2_dt)

                metrics = evaluate_dual_completion(
                    aux_r2, main_r2, test_code, entry_point, prompt
                )
                sample_metrics.append(metrics)

        per_sample_gen_metrics.append(sample_metrics)

    ks = [1, 3, 5, 10]
    metrics = compute_pass_at_k(per_sample_gen_metrics, ks)
    avg_resp = sum(avg_times) / len(avg_times) if avg_times else 0.0

    record = {
        "script": "multi_agent_tti",
        "dataset_input": dataset_name,
        "dataset": f"{dataset_name}:test[:{len(samples)}]",
        "aux_model": args.aux_model,
        "main_model": args.main_model,
        "tti_mode": mode,
        "tti_rounds": 2 if mode == "one_round_discussion" else 1,
        "samples": len(samples),
        "generations": args.generations,
        "k_values": ks,
        "metrics": metrics,
        "avg_response_time": avg_resp,
        "total_output_tokens": int(total_output_tokens),
        "avg_pass_rate": sum(m.get("passed_tests", 0) / (m.get("total_tests", 1) or 1)
                              for gens in per_sample_gen_metrics for m in gens) / max(1, sum(len(g) for g in per_sample_gen_metrics)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "no_job_id"),
        "timestamp": time.time(),
    }

    append_result_jsonl(args.result_json, record)


if __name__ == "__main__":
    main()

