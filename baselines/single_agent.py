import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project-local imports
from rewards.code_utils import (
    TimeoutException,
    check_function_definition,
    check_syntax,
    cleanup_code,
    extract_imports_from_prompt,
    extract_specific_function,
    extract_test_cases,
)

# Reuse the exact training prompt for single-agent
from train_grpo import (
    complete_function_formatter as sa_formatter,
)

# External prompt composition for 2-turn single-agent baselines
import external as external_ctx
from external import get_external_transition


def _normalize_prompt(p: str) -> str:
    return " ".join((p or "").split()).strip()


def _make_sliced_assert_tests(test_code: str, n: int) -> str:
    """Slice evaluation asserts for sandbox context prompts (external modes).

    n > 0: keep first n asserts; n < 0: keep last |n| asserts; n == 0/None: keep all.
    This affects only prompt construction for turn-2, not final evaluation.
    """
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
             temperature: float = 0.7, top_p: float = 0.9) -> Tuple[str, int, int, float]:
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


def evaluate_single_completion(
    completion: str,
    test_code: str,
    entry_point: str,
    prompt: str = "",
    per_test_timeout: int = 10,
) -> Dict[str, Any]:
    """Evaluate single-agent completion against extracted tests.

    Returns a metrics dict containing pass/fail details and counts.
    """
    metrics = {
        "extraction_successful": False,
        "syntax_correct": False,
        "test_results": [],
        "passed_tests": 0,
        "total_tests": 0,
        "timeout_num": 0,
        "execution_error": False,
        "is_correct": False,
    }

    if not completion or not completion.strip():
        return metrics

    imports = extract_imports_from_prompt(prompt)
    completion_clean = completion

    func_defined, _ = check_function_definition(
        completion_clean, entry_point, f"Function ({entry_point})"
    )
    metrics["extraction_successful"] = bool(func_defined)
    if not func_defined:
        return metrics

    combined_code = (imports + "\n" + completion_clean) if imports else completion_clean

    syntax_ok, _ = check_syntax(combined_code, "Candidate code")
    metrics["syntax_correct"] = bool(syntax_ok)
    if not syntax_ok:
        return metrics

    test_cases_list = extract_test_cases(test_code, entry_point)
    metrics["total_tests"] = len(test_cases_list)
    if not test_cases_list:
        return metrics

    timeout_count = 0
    passed_tests = 0
    results = []
    try:
        exec_globals = {}
        exec(combined_code, exec_globals)
        if entry_point not in exec_globals:
            metrics["execution_error"] = True
            return metrics

        for test_case in test_cases_list:
            if timeout_count >= 3:
                # fill remaining as False
                remaining = len(test_cases_list) - len(results)
                results.extend([False] * remaining)
                break
            try:
                import signal

                def _handler(signum, frame):
                    raise TimeoutException("timeout")

                signal.signal(signal.SIGALRM, _handler)
                signal.alarm(per_test_timeout)
                exec(test_case, exec_globals)
                signal.alarm(0)
                passed_tests += 1
                results.append(True)
            except TimeoutException:
                try:
                    import signal

                    signal.alarm(0)
                except Exception:
                    pass
                timeout_count += 1
                results.append(False)
            except Exception:
                try:
                    import signal

                    signal.alarm(0)
                except Exception:
                    pass
                results.append(False)
    except Exception:
        metrics["execution_error"] = True
        results = [False] * len(test_cases_list)

    metrics["passed_tests"] = passed_tests
    metrics["timeout_num"] = timeout_count
    metrics["test_results"] = results
    if metrics["total_tests"] > 0:
        metrics["is_correct"] = (passed_tests == metrics["total_tests"])
    return metrics


def compute_pass_at_k(sample_metrics: List[List[Dict[str, Any]]], ks: List[int]) -> Dict[str, float]:
    """Compute Pass@k across samples from per-generation metrics.

    For each sample, check if any of the first k generations pass all tests.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Unified single-agent baselines")
    parser.add_argument("--dataset", required=True, help="HF dataset id (e.g., openai/openai_humaneval or OpenMLRL/CoopHumanEval)")
    parser.add_argument("--model", required=True, help="Model name (HF id)")
    parser.add_argument("--num-turns", type=int, default=1, choices=[1, 2], help="Number of turns (1 or 2)")
    parser.add_argument("--external-mode", default=None, help="External mode for num_turns=2: plain|level_feedback|expert_edits")
    parser.add_argument("--expert-model", default="deepseek-coder", help="Expert model used by external transitions")
    parser.add_argument("--sandbox-slice", default="1", help="Slice asserts used in sandbox prompts: integer N, 0/all")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--generations", type=int, default=3, help="Generations per sample")
    parser.add_argument("--result-json", required=True, help="Output JSONL path to append a summary line")
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name = args.model
    num_turns = int(args.num_turns)

    t0 = time.time()

    # Load data
    test_data = load_dataset(dataset_name, split="test")
    total = len(test_data)
    start_idx = max(0, total - args.samples)
    samples = test_data.select(range(start_idx, total))

    # Register context for external modes (prompt -> entry_point/tests)
    context_map: Dict[str, Any] = {}

    # Parse sandbox slice
    _sv_raw = str(args.sandbox_slice).strip().lower()
    if _sv_raw == "all" or _sv_raw == "0":
        sandbox_slice = 0
    elif _sv_raw.lstrip("-").isdigit():
        sandbox_slice = int(_sv_raw)
    else:
        sandbox_slice = 1

    for item in samples:
        key = _normalize_prompt(item.get("prompt", ""))
        if key and key not in context_map:
            tests_eval = item.get("test", "")
            tests_sandbox = (
                _make_sliced_assert_tests(tests_eval, sandbox_slice)
                if sandbox_slice not in (None, 0)
                else tests_eval
            )
            context_map[key] = {
                "entry_point": item.get("entry_point", ""),
                "tests_eval": tests_eval,
                "tests_sandbox": tests_sandbox,
            }

    def _resolver(prompt: str):
        return context_map.get(_normalize_prompt(prompt))

    external_ctx.set_context_resolver(_resolver)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_name)

    # Storage
    per_sample_gen_metrics: List[List[Dict[str, Any]]] = []
    avg_resp_times: List[float] = []
    total_output_tokens = 0

    for sample in samples:
        prompt = sample["prompt"]
        entry_point = sample.get("entry_point", "")
        test_code_full = sample.get("test", "")

        # Turn 1 prompt using training-consistent formatter
        first_turn_prompt = sa_formatter({"prompt": prompt, "entry_point": entry_point})

        # Turn 1 generations (used if single-turn; also used for selecting seed for turn-2)
        turn1_completions: List[str] = []
        turn1_metrics: List[Dict[str, Any]] = []

        for _ in range(args.generations):
            resp, in_tok, out_tok, dt = generate(
                model, tokenizer, device, first_turn_prompt
            )
            total_output_tokens += out_tok
            avg_resp_times.append(dt)

            # Extract the specific function from response
            func_code = extract_specific_function(cleanup_code(resp), entry_point) or resp

            metrics = evaluate_single_completion(
                func_code, test_code_full, entry_point, prompt
            )

            turn1_completions.append(func_code)
            turn1_metrics.append(metrics)

        # If single-turn: record metrics and continue
        if num_turns == 1:
            per_sample_gen_metrics.append(turn1_metrics)
            continue

        # Multi-turn (2): build turn-2 prompt using external transitions
        if not args.external_mode:
            raise ValueError("--external-mode is required when --num-turns=2")

        # Select the best previous completion (highest passed_tests; tie by first)
        best_idx = 0
        best_score = -1
        for i, m in enumerate(turn1_metrics):
            score = m.get("passed_tests", 0)
            if score > best_score:
                best_score = score
                best_idx = i

        selected_main = turn1_completions[best_idx]

        # Compose turn-2 prompt using external transition (single-agent)
        next_prompts = get_external_transition(
            prompt=prompt,
            agent_completions=[selected_main],
            num_agents=1,
            mode=args.external_mode,
            expert_model=args.expert_model,
            original_prompt=True,
            previous_response=True,
        )
        second_turn_prompt = next_prompts[0] if isinstance(next_prompts, (list, tuple)) else str(next_prompts)

        # Turn 2 generations and evaluation
        turn2_metrics: List[Dict[str, Any]] = []
        for _ in range(args.generations):
            resp2, in_tok2, out_tok2, dt2 = generate(
                model, tokenizer, device, second_turn_prompt
            )
            total_output_tokens += out_tok2
            avg_resp_times.append(dt2)
            func_code2 = extract_specific_function(cleanup_code(resp2), entry_point) or resp2
            m2 = evaluate_single_completion(
                func_code2, test_code_full, entry_point, prompt
            )
            turn2_metrics.append(m2)

        # For 2-turn, we evaluate based on second turn only
        per_sample_gen_metrics.append(turn2_metrics)

    # Aggregate metrics
    ks = [1, 3, 5, 10]
    metrics = compute_pass_at_k(per_sample_gen_metrics, ks)
    avg_resp = sum(avg_resp_times) / len(avg_resp_times) if avg_resp_times else 0.0

    record = {
        "script": "single_agent",
        "dataset_input": dataset_name,
        "dataset": f"{dataset_name}:test[:{len(samples)}]",
        "model": model_name,
        "num_turns": num_turns,
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
    if num_turns == 2:
        record.update({
            "external_mode": args.external_mode,
            "sandbox_slice": str(args.sandbox_slice),
        })

    append_result_jsonl(args.result_json, record)


if __name__ == "__main__":
    main()
