"""
Single-agent baselines with optional two-turn external transitions (TTI).

Usage examples:

  # Single-turn baseline (HumanEval)
  python single_agent.py --dataset humaneval --model Qwen/Qwen2.5-Coder-7B \
    --samples 10 --generations 1

  # Two-turn baseline with external transitions (HumanEval)
  python single_agent.py --dataset humaneval --model Qwen/Qwen2.5-Coder-7B \
    --samples 10 --generations 1 --num-turns 2 \
    --external-mode level_feedback --sandbox-slice 1

Notes:
- When --num-turns=2, this script constructs next-turn prompts using the
  LLM_Collab_Code_Generation.external.get_external_transition API, similar
  to how training scripts set up external transitions in train_grpo.py.
- Rewards/metrics are computed via rewards.code_rewards.execution_reward_aux
  (in single-agent mode, aux is empty), keeping evaluation light-weight.
"""

import argparse
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Dict, List, Optional
import json
import time as _time
import fcntl

# Heavy deps are imported lazily at runtime

# Local modules


def extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    """Extract function parameters from a Python function signature in the prompt."""
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def single_turn_formatter(example: Dict[str, Any]) -> str:
    """Formatter for single-turn single-agent function completion prompts."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    params = extract_function_params_from_prompt(prompt)
    params_str = ", ".join(params)

    return (
        f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Do NOT include any text before or after the function
- Do NOT include test cases or example usage
- Implement ONLY the '{entry_point}' function as specified
- Make sure your solution is complete and handles all cases

Your output should follow this format:

def {entry_point}({params_str}):
    # your function code here
    return result
"""
    )


def setup_external_context(
    ds, sandbox_slice: Optional[int] = 1
):
    """Register dataset-aware context resolver for external transitions.

    Provides for each prompt:
      - entry_point
      - tests_eval (full eval tests)
      - tests_sandbox (optionally sliced asserts for feedback-type modes)

    sandbox_slice semantics (matching train_grpo.py):
      - int N > 0: keep first N asserts in check(candidate)
      - 0/None/'all': keep all eval asserts
      - negative values: keep last |N| asserts
    """

    def _normalize_prompt(p: str) -> str:
        return " ".join((p or "").split()).strip()

    def _make_sliced_assert_tests(test_code: str, n: Optional[int]) -> str:
        if not isinstance(test_code, str) or not test_code.strip():
            return test_code
        if n is None or n == 0:
            return test_code
        # Parse and keep first/last n asserts under def check(candidate)
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

    context_map: Dict[str, Dict[str, Any]] = {}
    for item in ds:
        key = _normalize_prompt(item.get("prompt", ""))
        if not key or key in context_map:
            continue
        tests_eval = item.get("test", "")
        tests_sbx = _make_sliced_assert_tests(tests_eval, sandbox_slice)
        context_map[key] = {
            "entry_point": item.get("entry_point", ""),
            "tests_eval": tests_eval,
            "tests_sandbox": tests_sbx,
        }

    def _resolver(p: str) -> Optional[Dict[str, Any]]:
        return context_map.get(_normalize_prompt(p))

    # Lazy import to avoid heavy imports at module import time
    import external as external_ctx  # type: ignore
    external_ctx.set_context_resolver(_resolver)


class SingleAgentRunner:
    """Lightweight single-agent generator using HF Transformers."""

    def __init__(self, model_name: str, device: str = "auto"):
        # Lazy imports to avoid loading heavy deps at import time
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure model is on the requested device (cpu/cuda)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
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




def _run_single_tests(main_func_code: str, prompt: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    """Execute tests for a single-agent completion and return pass metrics."""
    from rewards.code_utils import (
        extract_imports_from_prompt,
        extract_specific_function,
        cleanup_code,
        check_syntax,
        extract_test_cases,
        TimeoutException,
        timeout_handler,
    )
    import signal

    metrics = {
        "passed_tests": 0,
        "total_tests": 0,
        "timeouts": 0,
        "is_correct": False,
    }

    if not main_func_code:
        return metrics

    imports = extract_imports_from_prompt(prompt)
    cleaned = cleanup_code(main_func_code)
    func_only = extract_specific_function(cleaned or main_func_code, entry_point) or cleaned
    combined = (imports + "\n\n" + func_only) if imports else func_only

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
                # failed test
                continue
    except Exception:
        # loading failed
        pass

    metrics["passed_tests"] = passed
    metrics["timeouts"] = timeouts
    metrics["is_correct"] = (metrics["total_tests"] > 0 and passed == metrics["total_tests"]) 
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Single-agent baselines (concise): pass@k, timing, tokens, avg pass rate")
    parser.add_argument("--dataset", default="humaneval", help="Dataset type or HF repo name (e.g., humaneval | coophumaneval | OpenMLRL/CoopHumanEval)")
    parser.add_argument("--model", required=True, help="HF model name for the single agent")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device selection")
    parser.add_argument("--samples", type=int, default=31, help="Number of samples to evaluate (used when --hf-split is not set)")
    parser.add_argument("--hf-split", type=str, default=None, help="HuggingFace split expression (e.g., test[:16], test[133:]) overrides default split and --samples")
    parser.add_argument("--generations", type=int, default=1, help="Number of completions per sample")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10], help="k values for pass@k")
    parser.add_argument("--num-turns", type=int, default=1, help="Number of turns (1 or 2)")
    parser.add_argument("--result-json", type=str, default=None, help="Append a JSON line summary to this file")

    # External mode args (only used when num_turns=2)
    parser.add_argument("--external-mode", default="level_feedback", choices=["expert_edits", "level_feedback", "level_passed", "passed", "plain"], help="External transition mode for turn 2")
    parser.add_argument("--expert-model", default="deepseek-coder", help="LLM used by expert_edits mode")
    parser.add_argument("--sandbox-slice", default="1", help="Number of asserts to keep for feedback modes: int, 0/'all', negative for last asserts")
    parser.add_argument("--no-original-prompt", action="store_true", help="Do not include original prompt in turn-2 context")
    parser.add_argument("--no-previous-response", action="store_true", help="Do not include previous response in turn-2 context")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for reward/external modules")

    args = parser.parse_args()

    # Load dataset split (supports tokens or direct HF repo names)
    ds_arg = (args.dataset or "").strip()
    # Default splits aligned with configs: use eval splits by default
    # HE:  test[:32]; CHE: test[:16]; MBPP: test[:15]
    if ds_arg.lower() == "humaneval":
        ds_name, split = "openai/openai_humaneval", (args.hf_split or "test[:32]")
    elif ds_arg.lower() == "coophumaneval":
        ds_name, split = "OpenMLRL/CoopHumanEval", (args.hf_split or "test[:16]")
    elif ds_arg.lower() == "mbpp":
        ds_name, split = "OpenMLRL/MBPP", (args.hf_split or "test[:15]")
    else:
        # Treat as full HF repo path
        ds_name = ds_arg
        if args.hf_split:
            split = args.hf_split
        else:
            lname = ds_name.lower()
            # Heuristics for known datasets when a direct HF repo path is provided
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

    # External verbosity toggles
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

    # Prepare single-turn prompt formatter
    formatter = single_turn_formatter

    # Prepare external context if 2-turns
    is_multi_turn = args.num_turns > 1
    sandbox_slice: Optional[int]
    if isinstance(args.sandbox_slice, str):
        val = args.sandbox_slice.strip().lower()
        if val == "all":
            sandbox_slice = 0
        elif val.lstrip("-").isdigit():
            sandbox_slice = int(val)
        else:
            sandbox_slice = None
    else:
        sandbox_slice = int(args.sandbox_slice)

    if is_multi_turn:
        setup_external_context(test_samples, sandbox_slice)

    # Load model
    runner = SingleAgentRunner(model_name=args.model, device=args.device)

    # Iterate samples: generate multiple completions per sample and evaluate
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
        sample_correct_flags: List[bool] = []
        # Generate G completions for this sample
        for g in range(args.generations):
            # Turn 1
            turn1_prompt = formatter(example)
            t1_text, t1_in, t1_out, t1_time = runner.generate(turn1_prompt)
            out_text = t1_text
            gen_time = t1_time
            total_output_tokens += t1_out

            # Turn 2 (optional)
            if is_multi_turn:
                from external import get_external_transition  # lazy import
                next_prompts = get_external_transition(
                    prompt=example["prompt"],
                    agent_completions=[t1_text],
                    num_agents=1,
                    mode=args.external_mode,
                    expert_model=args.expert_model,
                    original_prompt=not args.no_original_prompt,
                    previous_response=not args.no_previous_response,
                )
                turn2_prompt = next_prompts[0] if isinstance(next_prompts, (list, tuple)) else str(next_prompts)
                t2_text, t2_in, t2_out, t2_time = runner.generate(turn2_prompt)
                out_text = t2_text
                gen_time += t2_time
                total_output_tokens += t2_out

            # Evaluate
            m = _run_single_tests(out_text, example.get("prompt", ""), example.get("test", ""), example.get("entry_point", ""))
            all_times.append(gen_time)
            sample_correct_flags.append(bool(m["is_correct"]))
            if m["total_tests"] > 0:
                per_completion_pass_rates.append(m["passed_tests"] / m["total_tests"]) 

        # pass@k per sample using HumanEval estimator
        m = sum(1 for b in sample_correct_flags if b)
        n = len(sample_correct_flags)
        for k in args.k_values:
            per_sample_pass_at_k[k].append(passk(m, n, k))

    # Aggregate
    avg_resp_time = float(np.mean(all_times)) if all_times else 0.0
    avg_pass_rate = float(np.mean(per_completion_pass_rates)) if per_completion_pass_rates else 0.0
    pass_at_k_summary = {f"pass@{k}": float(np.mean(v)) if v else 0.0 for k, v in per_sample_pass_at_k.items()}

    # Print concise summary
    print("\n" + "=" * 60)
    print("Single-agent baseline (concise)")
    print(f"Model: {args.model}")
    print(f"Dataset: {ds_name}:{split}")
    print(f"Turns: {args.num_turns}")
    if is_multi_turn:
        print(f"External: mode={args.external_mode}, sandbox_slice={args.sandbox_slice}")
    print(f"Samples: {len(test_samples)} | Generations per sample: {args.generations}")
    print("Metrics:")
    print("  " + ", ".join([f"{k}={v:.3f}" for k, v in pass_at_k_summary.items()]))
    print(f"  avg_response_time={avg_resp_time:.2f}s")
    print(f"  total_output_tokens={total_output_tokens}")
    print(f"  avg_pass_rate={avg_pass_rate:.3f}")

    # Optionally append JSONL summary (safe append with file lock)
    if args.result_json:
        summary: Dict[str, Any] = {
            "script": "single_agent",
            "dataset_input": args.dataset,
            "dataset": f"{ds_name}:{split}",
            "model": args.model,
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
        # Remove None fields
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
