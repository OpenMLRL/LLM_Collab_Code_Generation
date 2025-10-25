"""
Multi-agent "ours" baseline with optional two-turn external transitions.

This script runs a two-agent pipeline where:
  - Turn 1: Aux agent writes `aux(...)`; Main agent writes the target function (may call aux).
  - Turn 2 (optional): Prompts are revised via external.get_external_transition using a
    chosen mode (level_feedback, expert_edits, etc.), then both agents regenerate.

Rewards/metrics are computed via rewards.code_rewards.execution_reward_aux.

Examples:
  # Single-turn (no external), same model for both agents
  python multi_agent_ours.py --dataset humaneval --model Qwen/Qwen2.5-Coder-7B \
    --samples 10

  # Two-turn with external transitions
  python multi_agent_ours.py --dataset humaneval --model Qwen/Qwen2.5-Coder-7B \
    --samples 10 --num-turns 2 --external-mode level_feedback --sandbox-slice 1
"""

import argparse
import re
from typing import Any, Dict, List, Optional, Tuple
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time as _time
import fcntl



def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def aux_round1_formatter(prompt: str, entry_point: str) -> str:
    params = extract_function_params_from_prompt(prompt)
    params_str = ", ".join(params)
    return (
        f"""Create a helper function for this coding problem.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Create a helper function named 'aux' that can assist the main function
- The aux function MUST use the same parameters as the main function: ({params_str})
- The function should return useful data for solving the problem

Your output should follow this format:

def aux({params_str}):
    # your function code here
    return result
"""
    )


def main_round1_formatter(prompt: str, entry_point: str) -> str:
    params = extract_function_params_from_prompt(prompt)
    params_str = ", ".join(params)
    return (
        f"""Solve this coding problem by implementing the required function.

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
- You can call aux() to assign value to a variable within your function

Your output should follow this format:

def {entry_point}({params_str}):
    # your function code here
    return result
"""
    )


def setup_external_context(ds, sandbox_slice: Optional[int] = 1):
    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    def _make_sliced_assert_tests(test_code: str, n: Optional[int]) -> str:
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
        selected = asserts[:n] if (n is not None and n > 0) else asserts[n:]
        for a in selected:
            new_parts.append(f"    {a}")
        return "\n".join(new_parts) + "\n"

    cmap: Dict[str, Dict[str, Any]] = {}
    for item in ds:
        key = _normalize_prompt(item.get("prompt", ""))
        if not key or key in cmap:
            continue
        tests_eval = item.get("test", "")
        tests_sbx = _make_sliced_assert_tests(tests_eval, sandbox_slice)
        cmap[key] = {
            "entry_point": item.get("entry_point", ""),
            "tests_eval": tests_eval,
            "tests_sandbox": tests_sbx,
        }

    def _resolver(p: str) -> Optional[Dict[str, Any]]:
        return cmap.get(_normalize_prompt(p))

    # Lazy import
    import external as external_ctx  # type: ignore
    external_ctx.set_context_resolver(_resolver)


class Agent:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure model is on the requested device (cpu/cuda)
        self.model = self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.95):
        import time
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_tokens = len(generated_tokens)
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text, input_tokens, output_tokens, elapsed


def _run_aux_main_tests(aux_code: str, main_code: str, prompt: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    from rewards.code_utils import (
        extract_imports_from_prompt,
        extract_specific_function,
        cleanup_code,
        check_syntax,
        extract_test_cases,
        concatenate_functions,
        TimeoutException,
        timeout_handler,
    )
    import signal

    metrics = {"passed_tests": 0, "total_tests": 0, "timeouts": 0, "is_correct": False}
    imports = extract_imports_from_prompt(prompt)
    aux_clean = cleanup_code(aux_code)
    # Prefer the specifically-named 'aux' function; if absent (e.g., expert edits rename helper),
    # fall back to including the entire cleaned aux completion to preserve the helper definition.
    aux_func = extract_specific_function(aux_clean or aux_code, "aux")
    main_func = extract_specific_function(cleanup_code(main_code) or main_code, entry_point)
    combined = concatenate_functions(aux_func or aux_clean, main_func, imports)

    ok, _ = check_syntax(combined, "Combined code")
    if not ok:
        return metrics

    tests = extract_test_cases(test_code, entry_point)
    if not tests:
        return metrics
    metrics["total_tests"] = len(tests)

    TEST_TIMEOUT = 10
    MAX_TIMEOUTS = 3
    passed = 0
    timeouts = 0
    try:
        env = {}
        exec(combined, env)
        if entry_point not in env:
            return metrics
        for t in tests:
            if timeouts >= MAX_TIMEOUTS:
                break
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TEST_TIMEOUT)
                exec(t, env)
                passed += 1
                signal.alarm(0)
            except TimeoutException:
                signal.alarm(0)
                timeouts += 1
            except Exception:
                signal.alarm(0)
                continue
    except Exception:
        pass
    metrics["passed_tests"] = passed
    metrics["timeouts"] = timeouts
    metrics["is_correct"] = (metrics["total_tests"] > 0 and passed == metrics["total_tests"]) 
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-agent ours (concise): pass@k, timing, tokens, avg pass rate")
    parser.add_argument("--dataset", default="humaneval", help="Benchmark dataset or HF repo name (e.g., humaneval | coophumaneval | OpenMLRL/CoopHumanEval)")
    parser.add_argument("--model", help="Use the same model for AUX and MAIN agents")
    parser.add_argument("--aux-model", help="Aux agent model name (overrides --model)")
    parser.add_argument("--main-model", help="Main agent model name (overrides --model)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--samples", type=int, default=31, help="Number of samples to evaluate (used when --hf-split is not set)")
    parser.add_argument("--hf-split", type=str, default=None, help="HuggingFace split expression (e.g., test[:16])")
    parser.add_argument("--generations", type=int, default=1, help="Generations per sample")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10], help="k values for pass@k")
    parser.add_argument("--num-turns", type=int, default=1, help="Number of turns (1 or 2)")
    parser.add_argument("--result-json", type=str, default=None, help="Append a JSON line summary to this file")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Top-p nucleus sampling")

    # External options (turn-2 only)
    parser.add_argument("--external-mode", default="level_feedback", choices=["expert_edits", "level_feedback", "level_passed", "passed", "plain"], help="External transition mode")
    parser.add_argument("--expert-model", default="deepseek-coder", help="LLM used by expert_edits mode")
    parser.add_argument("--sandbox-slice", default="1", help="Number of asserts to keep for feedback modes: int, 0/'all', negative for last asserts")
    parser.add_argument("--no-original-prompt", action="store_true", help="Do not include original prompt in turn-2 context")
    parser.add_argument("--no-previous-response", action="store_true", help="Do not include previous response in turn-2 context")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for reward/external modules")

    args = parser.parse_args()

    # Resolve models
    aux_model = args.aux_model or args.model
    main_model = args.main_model or args.model
    if (aux_model is None) or (main_model is None):
        raise SystemExit("Please provide --model or both --aux-model and --main-model.")

    # Dataset (supports tokens or direct HF repo names)
    ds_arg = (args.dataset or "").strip()
    # Default splits aligned with configs (use eval splits)
    # HE:  test[:32]; CHE: test[:16]; MBPP: test[:15]
    if ds_arg.lower() == "humaneval":
        ds_name, split = "openai/openai_humaneval", (args.hf_split or "test[:32]")
    elif ds_arg.lower() == "coophumaneval":
        ds_name, split = "OpenMLRL/CoopHumanEval", (args.hf_split or "test[:16]")
    elif ds_arg.lower() == "mbpp":
        ds_name, split = "OpenMLRL/MBPP", (args.hf_split or "test[:15]")
    else:
        ds_name = ds_arg
        if args.hf_split:
            split = args.hf_split
        else:
            lname = ds_name.lower()
            if ("humaneval" in lname) and ("coop" not in lname):
                split = "test[:32]"  # HE eval split
            elif ("coop" in lname) or ("coophumaneval" in lname):
                split = "test[:16]"  # CHE eval split
            elif ("mbpp" in lname):
                split = "test[:15]"  # MBPP eval split
            else:
                split = "test"
    try:
        from datasets import load_dataset
        test_data = load_dataset(ds_name, split=split)
    except Exception as e:
        print(f"Failed to load dataset {ds_name}:{split}: {e}")
        return
    if args.hf_split:
        test_samples = test_data
        print(f"Evaluating {len(test_samples)} samples from {ds_name}:{split}")
    else:
        total = len(test_data)
        start_idx = max(0, total - args.samples)
        test_samples = test_data.select(range(start_idx, total))
        print(f"Evaluating {len(test_samples)} samples from {ds_name}:{split} (indices {start_idx}-{total-1})")

    # Verbosity
    try:
        import rewards.code_rewards as code_rewards
        code_rewards.VERBOSE = bool(args.verbose)
    except Exception:
        pass
    try:
        import external as external_mod
        external_mod.VERBOSE = bool(args.verbose)
    except Exception:
        pass

    # External context
    is_multi_turn = args.num_turns > 1
    if isinstance(args.sandbox_slice, str):
        ssv = args.sandbox_slice.strip().lower()
        if ssv == "all":
            sandbox_slice = 0
        elif ssv.lstrip("-").isdigit():
            sandbox_slice = int(ssv)
        else:
            sandbox_slice = None
    else:
        sandbox_slice = int(args.sandbox_slice)
    if is_multi_turn:
        setup_external_context(test_samples, sandbox_slice)  # register resolver for external

    # Agents
    aux_agent = Agent(aux_model, args.device)
    main_agent = Agent(main_model, args.device)

    import numpy as np
    all_times: List[float] = []
    total_output_tokens = 0
    per_sample_pass_at_k: Dict[int, List[float]] = {k: [] for k in args.k_values}
    per_completion_pass_rates: List[float] = []
    from math import comb

    def passk(m: int, n: int, k: int) -> float:
        if n <= 0 or k <= 0:
            return 0.0
        if n < k:
            return 1.0 if m > 0 else 0.0
        if m <= 0:
            return 0.0
        return 1.0 - (comb(n - m, k) / comb(n, k))

    for example in test_samples:
        prompt = example["prompt"]
        entry_point = example["entry_point"]
        test_code = example.get("test", "")

        sample_correct_flags: List[bool] = []

        for g in range(args.generations):
            # Turn 1
            aux_prompt_1 = aux_round1_formatter(prompt, entry_point)
            main_prompt_1 = main_round1_formatter(prompt, entry_point)
            aux_out_1, a1_in, a1_out, a1_time = aux_agent.generate(aux_prompt_1, temperature=args.temperature, top_p=args.top_p)
            main_out_1, m1_in, m1_out, m1_time = main_agent.generate(main_prompt_1, temperature=args.temperature, top_p=args.top_p)
            gen_time = a1_time + m1_time
            total_output_tokens += (a1_out + m1_out)

            aux_final, main_final = aux_out_1, main_out_1
            if is_multi_turn:
                from external import get_external_transition  # lazy import
                next_prompts = get_external_transition(
                    prompt=prompt,
                    agent_completions=[aux_out_1, main_out_1],
                    num_agents=2,
                    mode=args.external_mode,
                    expert_model=args.expert_model,
                    original_prompt=not args.no_original_prompt,
                    previous_response=not args.no_previous_response,
                )
                if isinstance(next_prompts, (list, tuple)) and len(next_prompts) == 2:
                    aux_prompt_2, main_prompt_2 = next_prompts
                else:
                    aux_prompt_2, main_prompt_2 = aux_prompt_1, (next_prompts[0] if isinstance(next_prompts, (list, tuple)) else str(next_prompts))
                aux_final, a2_in, a2_out, a2_time = aux_agent.generate(aux_prompt_2, temperature=args.temperature, top_p=args.top_p)
                main_final, m2_in, m2_out, m2_time = main_agent.generate(main_prompt_2, temperature=args.temperature, top_p=args.top_p)
                gen_time += (a2_time + m2_time)
                total_output_tokens += (a2_out + m2_out)

            # Evaluate
            m = _run_aux_main_tests(aux_final, main_final, prompt, test_code, entry_point)
            all_times.append(gen_time)
            sample_correct_flags.append(bool(m["is_correct"]))
            if m["total_tests"] > 0:
                per_completion_pass_rates.append(m["passed_tests"] / m["total_tests"]) 

        # pass@k per sample
        m = sum(1 for b in sample_correct_flags if b)
        n = len(sample_correct_flags)
        for k in args.k_values:
            per_sample_pass_at_k[k].append(passk(m, n, k))

    # Aggregate
    avg_resp_time = float(np.mean(all_times)) if all_times else 0.0
    avg_pass_rate = float(np.mean(per_completion_pass_rates)) if per_completion_pass_rates else 0.0
    pass_at_k_summary = {f"pass@{k}": float(np.mean(v)) if v else 0.0 for k, v in per_sample_pass_at_k.items()}

    print("\n" + "=" * 60)
    print("Multi-agent ours baseline (concise)")
    print(f"Dataset: {ds_name}:{split}")
    print(f"Aux model: {aux_model}")
    print(f"Main model: {main_model}")
    print(f"Turns: {args.num_turns}")
    if is_multi_turn:
        print(f"External: mode={args.external_mode}, sandbox_slice={args.sandbox_slice}")
    print(f"Samples: {len(test_samples)} | Generations per sample: {args.generations}")
    print("Metrics:")
    print("  " + ", ".join([f"{k}={v:.3f}" for k, v in pass_at_k_summary.items()]))
    print(f"  avg_response_time={avg_resp_time:.2f}s")
    print(f"  total_output_tokens={total_output_tokens}")
    print(f"  avg_pass_rate={avg_pass_rate:.3f}")

    # Optionally append JSONL summary
    if args.result_json:
        summary: Dict[str, Any] = {
            "script": "multi_agent_ours",
            "dataset_input": args.dataset,
            "dataset": f"{ds_name}:{split}",
            "aux_model": aux_model,
            "main_model": main_model,
            "num_turns": args.num_turns,
            "external_mode": args.external_mode if (args.num_turns > 1) else None,
            "sandbox_slice": args.sandbox_slice if (args.num_turns > 1) else None,
            "samples": len(test_samples),
            "generations": args.generations,
            "k_values": args.k_values,
            "metrics": pass_at_k_summary,
            "avg_response_time": avg_resp_time,
            "total_output_tokens": total_output_tokens,
            "avg_pass_rate": avg_pass_rate,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": _time.time(),
        }
        summary = {k: v for k, v in summary.items() if v is not None}
        try:
            os.makedirs(os.path.dirname(args.result_json), exist_ok=True) if os.path.dirname(args.result_json) else None
            with open(args.result_json, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(summary) + "\n")
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Failed to write summary JSONL to {args.result_json}: {e}")


if __name__ == "__main__":
    main()
