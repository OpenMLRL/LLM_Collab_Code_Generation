"""
Concise multi-agent TTI baselines: naive_concat, sequential_pipeline, one_round_discussion.

Outputs only: pass@k, avg response time, total generated tokens, avg pass rate.
"""

import argparse
import re
from typing import Any, Dict, List, Optional
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time as _time
import fcntl



def _extract_params(prompt: str) -> List[str]:
    m = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt or "")
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def fmt_aux(prompt: str) -> str:
    params = ", ".join(_extract_params(prompt))
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

def aux({params}):
    # your function code here
    return result
"""


def fmt_main_naive(prompt: str, entry_point: str) -> str:
    params = ", ".join(_extract_params(prompt))
    return f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux({params})

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function
- Implement ONLY the '{entry_point}' function as specified
- You call aux() to assign value to a variable within your function

Your output should follow this format:

def {entry_point}({params}):
    # your function code here
    return result
"""


def fmt_main_sequential(prompt: str, entry_point: str, aux_code: str) -> str:
    params = ", ".join(_extract_params(prompt))
    return f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to this helper function that has already been implemented:

{aux_code}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Do NOT redefine the aux() function (it's already provided above)
- Implement ONLY the '{entry_point}' function as specified
- You can call aux() to get useful data for solving the problem
- Make sure to use the aux() function effectively in your solution

Your output should follow this format:

def {entry_point}({params}):
    # your function code here
    return result
"""


def fmt_aux_round2(prompt: str, main_code: str) -> str:
    params = ", ".join(_extract_params(prompt))
    return f"""Improve your helper function based on how it's being used by the main function.

Problem:
{prompt}

The main function implementation from another agent:
{main_code}

IMPORTANT INSTRUCTIONS:
- Output ONLY the improved aux function code, no explanations
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Create an improved helper function named 'aux' that better assists the main function
- Look at how the main function uses (or could use) aux() and optimize accordingly
- The aux function should provide exactly what the main function needs

Your output should follow this format:

def aux({params}):
    # your improved function code here
    return result
"""


def fmt_main_round2(prompt: str, entry_point: str, aux_code: str) -> str:
    params = ", ".join(_extract_params(prompt))
    return f"""Improve your main function implementation based on the helper function provided.

Problem:
{prompt}

The auxiliary function implementation from another agent:
{aux_code}

IMPORTANT INSTRUCTIONS:
- Output ONLY the improved main function code, no explanations
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT redefine the aux() function (it's already provided above)
- Implement ONLY the improved '{entry_point}' function
- Make effective use of the aux() function based on what it provides
- Ensure your solution correctly uses the aux() function's return value

Your output should follow this format:

def {entry_point}({params}):
    # your improved function code here
    return result
"""


class Agent:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            )
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
    # Keep parity with training: extract only 'aux' and the exact entry_point
    aux_func = extract_specific_function(aux_clean or aux_code, "aux")
    main_func = extract_specific_function(cleanup_code(main_code) or main_code, entry_point)
    combined = concatenate_functions(aux_func, main_func, imports)

    # Optional debug; enable with EVAL_DEBUG=1
    try:
        import os as _os
        if str(_os.environ.get("EVAL_DEBUG", "")).lower() in ("1", "true", "yes"):
            if not aux_func:
                print("[DEBUG] aux_func is empty after extraction; showing aux_clean head:")
                print((aux_clean or aux_code)[:300])
            if not main_func:
                print(f"[DEBUG] main_func '{entry_point}' is empty after extraction; showing main_clean head:")
                _mc = cleanup_code(main_code) or main_code
                print((_mc)[:300])
    except Exception:
        pass

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
    parser = argparse.ArgumentParser(description="Multi-agent TTI baselines (concise): pass@k, timing, tokens, avg pass rate")
    parser.add_argument("--dataset", default="humaneval", help="Benchmark dataset or HF repo name (e.g., humaneval | coophumaneval | OpenMLRL/CoopHumanEval)")
    parser.add_argument("--mode", required=True, choices=["naive_concat", "sequential_pipeline", "one_round_discussion"], help="Interaction mode")
    parser.add_argument("--model", help="Use the same model for both agents")
    parser.add_argument("--aux-model", help="Auxiliary agent model name")
    parser.add_argument("--main-model", help="Main agent model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--samples", type=int, default=31, help="Number of samples to evaluate (used when --hf-split is not set)")
    parser.add_argument("--hf-split", type=str, default=None, help="HuggingFace split expression, e.g., test[:16]")
    parser.add_argument("--generations", type=int, default=1, help="Generations per sample")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10], help="k values for pass@k")
    parser.add_argument("--result-json", type=str, default=None, help="Append a JSON line summary to this file")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="Top-p nucleus sampling")

    args = parser.parse_args()

    aux_model = args.aux_model or args.model
    main_model = args.main_model or args.model
    if (aux_model is None) or (main_model is None):
        raise SystemExit("Please provide --model or both --aux-model and --main-model.")

    # Dataset (supports tokens or direct HF repo names)
    ds_arg = (args.dataset or "").strip()
    # Default splits aligned with training configs (use eval splits)
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
    else:
        total = len(test_data)
        start_idx = max(0, total - args.samples)
        test_samples = test_data.select(range(start_idx, total))

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
            # Round 1
            aux_p1 = fmt_aux(prompt)
            aux_out_1, a1_in, a1_out, a1_time = aux_agent.generate(aux_p1, temperature=args.temperature, top_p=args.top_p)

            if args.mode == "naive_concat":
                main_p1 = fmt_main_naive(prompt, entry_point)
            else:  # sequential_pipeline and discussion both use main that can call aux; only sequential sees code
                if args.mode == "sequential_pipeline":
                    main_p1 = fmt_main_sequential(prompt, entry_point, aux_out_1)
                else:
                    # discussion: round1 main doesn't see aux code (match existing he_discuss round1)
                    main_p1 = fmt_main_naive(prompt, entry_point)

            main_out_1, m1_in, m1_out, m1_time = main_agent.generate(main_p1, temperature=args.temperature, top_p=args.top_p)
            gen_time = a1_time + m1_time
            total_output_tokens += (a1_out + m1_out)

            aux_final, main_final = aux_out_1, main_out_1
            if args.mode == "one_round_discussion":
                # Round 2 exchange
                aux_p2 = fmt_aux_round2(prompt, main_out_1)
                main_p2 = fmt_main_round2(prompt, entry_point, aux_out_1)
                aux_final, a2_in, a2_out, a2_time = aux_agent.generate(aux_p2, temperature=args.temperature, top_p=args.top_p)
                main_final, m2_in, m2_out, m2_time = main_agent.generate(main_p2, temperature=args.temperature, top_p=args.top_p)
                gen_time += (a2_time + m2_time)
                total_output_tokens += (a2_out + m2_out)

            m = _run_aux_main_tests(aux_final, main_final, prompt, test_code, entry_point)
            all_times.append(gen_time)
            sample_correct_flags.append(bool(m["is_correct"]))
            if m["total_tests"] > 0:
                per_completion_pass_rates.append(m["passed_tests"] / m["total_tests"]) 

        m = sum(1 for b in sample_correct_flags if b)
        n = len(sample_correct_flags)
        for k in args.k_values:
            per_sample_pass_at_k[k].append(passk(m, n, k))

    avg_resp_time = float(np.mean(all_times)) if all_times else 0.0
    avg_pass_rate = float(np.mean(per_completion_pass_rates)) if per_completion_pass_rates else 0.0
    pass_at_k_summary = {f"pass@{k}": float(np.mean(v)) if v else 0.0 for k, v in per_sample_pass_at_k.items()}

    print("\n" + "=" * 60)
    print("Multi-agent TTI baseline (concise)")
    print(f"Dataset: {ds_name}:{split}")
    print(f"Aux model: {aux_model}")
    print(f"Main model: {main_model}")
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(test_samples)} | Generations per sample: {args.generations}")
    print("Metrics:")
    print("  " + ", ".join([f"{k}={v:.3f}" for k, v in pass_at_k_summary.items()]))
    print(f"  avg_response_time={avg_resp_time:.2f}s")
    print(f"  total_output_tokens={total_output_tokens}")
    print(f"  avg_pass_rate={avg_pass_rate:.3f}")

    # Optionally append JSONL summary
    if args.result_json:
        tti_rounds = 2 if args.mode == "one_round_discussion" else 1
        summary: Dict[str, Any] = {
            "script": "multi_agent_tti",
            "dataset_input": args.dataset,
            "dataset": f"{ds_name}:{split}",
            "aux_model": aux_model,
            "main_model": main_model,
            "tti_mode": args.mode,
            "tti_rounds": tti_rounds,
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
