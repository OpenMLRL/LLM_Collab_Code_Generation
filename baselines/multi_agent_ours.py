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
    cleanup_code,
    extract_specific_function,
)

from LLM_Collab_Code_Generation.train_magrpo import (
    aux_function_formatter as aux_formatter,
    main_function_formatter as main_formatter,
)

from LLM_Collab_Code_Generation.baselines.multi_agent_tti import (
    load_model_and_tokenizer,
    generate,
    evaluate_dual_completion,
    compute_pass_at_k,
)

import LLM_Collab_Code_Generation.external as external_ctx
from LLM_Collab_Code_Generation.external import get_external_transition


def _normalize_prompt(p: str) -> str:
    return " ".join((p or "").split()).strip()


def _make_sliced_assert_tests(test_code: str, n: int) -> str:
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


def append_result_jsonl(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Our multi-agent baselines (MAGRPO-style prompts + optional external turns)")
    parser.add_argument("--dataset", required=True, help="HF dataset id (HE/CHE)")
    parser.add_argument("--aux-model", required=True, help="Auxiliary model name (HF id)")
    parser.add_argument("--main-model", required=True, help="Main model name (HF id)")
    parser.add_argument("--num-turns", type=int, default=1, choices=[1, 2], help="Number of turns (1 or 2)")
    parser.add_argument("--external-mode", default=None, help="External mode for num_turns=2: plain|level_feedback|expert_edits")
    parser.add_argument("--expert-model", default="deepseek-coder", help="Expert model used by external transitions")
    parser.add_argument("--sandbox-slice", default="1", help="Slice asserts used in sandbox prompts: integer N, 0/all")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--generations", type=int, default=3, help="Generations per sample")
    parser.add_argument("--result-json", required=True, help="Output JSONL path to append a summary line")
    args = parser.parse_args()

    dataset_name = args.dataset
    num_turns = int(args.num_turns)

    # Data
    test_data = load_dataset(dataset_name, split="test")
    total = len(test_data)
    start_idx = max(0, total - args.samples)
    samples = test_data.select(range(start_idx, total))

    # Register context for external (prompt -> entry/tests)
    context_map: Dict[str, Any] = {}
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

        if num_turns == 1:
            # Single-turn MAGRPO-style prompts
            for _ in range(args.generations):
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

            per_sample_gen_metrics.append(sample_metrics)
            continue

        # Two-turn with external transitions
        if not args.external_mode:
            raise ValueError("--external-mode is required when --num-turns=2")

        # Turn 1: generate one pair to seed external prompts (use first generation to keep runtime bounded)
        aux_prompt_r1 = aux_formatter({"prompt": prompt, "entry_point": entry_point})
        aux_r1, a1_in, a1_out, a1_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt_r1)
        main_prompt_r1 = main_formatter({"prompt": prompt, "entry_point": entry_point})
        main_r1, m1_in, m1_out, m1_dt = generate(main_model, main_tok, main_dev, main_prompt_r1)

        aux_func_r1 = extract_specific_function(cleanup_code(aux_r1), "aux") or aux_r1
        main_func_r1 = extract_specific_function(cleanup_code(main_r1), entry_point) or main_r1

        # Compose second-turn prompts via external transitions
        next_prompts = get_external_transition(
            prompt=prompt,
            agent_completions=[aux_func_r1, main_func_r1],
            num_agents=2,
            mode=args.external_mode,
            expert_model=args.expert_model,
            original_prompt=True,
            previous_response=True,
        )
        if not isinstance(next_prompts, (list, tuple)) or len(next_prompts) != 2:
            raise RuntimeError("external transition did not return two prompts for agents")
        aux_prompt_r2, main_prompt_r2 = next_prompts[0], next_prompts[1]

        # Turn 2: generate multiple completions to evaluate
        for _ in range(args.generations):
            aux_r2, a2_in, a2_out, a2_dt = generate(aux_model, aux_tok, aux_dev, aux_prompt_r2)
            main_r2, m2_in, m2_out, m2_dt = generate(main_model, main_tok, main_dev, main_prompt_r2)

            total_output_tokens += (a1_out + m1_out + a2_out + m2_out)
            avg_times.append(a1_dt + m1_dt + a2_dt + m2_dt)

            metrics = evaluate_dual_completion(
                aux_r2, main_r2, test_code, entry_point, prompt
            )
            sample_metrics.append(metrics)

        per_sample_gen_metrics.append(sample_metrics)

    # Aggregate
    ks = [1, 3, 5, 10]
    metrics = compute_pass_at_k(per_sample_gen_metrics, ks)
    avg_resp = sum(avg_times) / len(avg_times) if avg_times else 0.0

    record = {
        "script": "multi_agent_ours",
        "dataset_input": dataset_name,
        "dataset": f"{dataset_name}:test[:{len(samples)}]",
        "aux_model": args.aux_model,
        "main_model": args.main_model,
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

