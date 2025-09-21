#!/usr/bin/env python3
"""
Generate and save prompts from external modes for num_agents=1 (sa) and 2 (ma).

Outputs two files under test/ only:
  - prompts_sa.txt  (single agent)
  - prompts_ma.txt  (multi agent)

Avoids network by stubbing expert_edits.add_expert_edits.
"""

import os
import sys
import argparse
from typing import Dict


def project_root() -> str:
    # This file lives at <repo_root>/LLM_Collaboration_with_MARL/test
    # We want sys.path to include <repo_root>/LLM_Collaboration_with_MARL
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )


def add_repo_to_path():
    root = project_root()
    # root here is <repo_root>/LLM_Collaboration_with_MARL
    if root not in sys.path:
        sys.path.insert(0, root)


def build_context(prompt: str) -> Dict[str, str]:
    entry_point = "add"
    # Minimal tests that the framework can parse (uses candidate -> replaced by entry_point)
    tests = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(-1, 1) == 0\n"
        "    assert candidate(0, 0) == 0\n"
    )
    return {"entry_point": entry_point, "tests_eval": tests, "tests_sandbox": tests}


def main():
    parser = argparse.ArgumentParser(description="Dump external prompts for 1/2 agents")
    parser.add_argument(
        "--real-expert",
        action="store_true",
        help="Force call real expert model for expert_edits (requires API key)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline stub for expert_edits (no network)",
    )
    parser.add_argument(
        "--expert-model",
        type=str,
        default=os.environ.get("EXPERT_MODEL", "deepseek-coder"),
        help="Expert model name for expert_edits",
    )
    args = parser.parse_args()

    add_repo_to_path()

    # Default to real expert calls unless offline is requested
    def _is_truthy(v: str) -> bool:
        return str(v).lower() in ("1", "true", "yes", "y")

    use_real_expert = True
    if _is_truthy(os.environ.get("OFFLINE", "0")) or args.offline:
        use_real_expert = False
    if _is_truthy(os.environ.get("REAL_EXPERT", "0")):
        use_real_expert = True
    if args.real_expert:
        use_real_expert = True

    # Ensure 'anthropic' is importable (not used unless that model is chosen)
    try:
        import anthropic  # type: ignore  # noqa: F401
    except Exception:
        import types

        if "anthropic" not in sys.modules:
            m = types.ModuleType("anthropic")

            class _Anthropic:
                def __init__(self, *a, **k):
                    pass

            m.Anthropic = _Anthropic
            sys.modules["anthropic"] = m

    # Stub openai only when not using real expert edits
    if not use_real_expert:
        import types

        if "openai" not in sys.modules:
            m2 = types.ModuleType("openai")

            class _OpenAI:
                def __init__(self, *a, **k):
                    pass

            m2.OpenAI = _OpenAI
            sys.modules["openai"] = m2

    # Import our local package (ensure we don't shadow by this test module name later)
    import importlib

    external = importlib.import_module("external")

    # Register a simple context resolver
    def resolver(p: str):
        return build_context(p)

    external.set_context_resolver(resolver)

    # Monkey patch expert_edits.add_expert_edits to avoid network calls
    if not use_real_expert:
        from external import expert_edits as ee

        def _stub_add_expert_edits(
            prompt: str, aux_completion: str, main_completion: str, **kwargs
        ):
            # Return deterministic edits; leave aux empty for single-agent compatibility
            return (
                prompt,
                "# AUX EDIT: (stub) helper not needed for this task",
                "# MAIN EDIT: (stub) handle edge cases and negative inputs",
            )

        ee.add_expert_edits = _stub_add_expert_edits  # type: ignore

    # Inputs
    original_prompt = "Write a function add(x: int, y: int) -> int that returns the sum of two integers."

    # Completions for num_agents=1 (main only)
    main_only_code = (
        "def add(x, y):\n" "    # simple implementation\n" "    return x + y\n"
    )

    # Completions for num_agents=2 (aux + main)
    aux_code = "def aux(x, y):\n" "    # helper returns sum\n" "    return x + y\n"
    main_code = "def add(x, y):\n" "    # use helper\n" "    return aux(x, y)\n"

    modes = ["expert_edits", "level_feedback", "level_passed", "passed", "plain"]

    out_dir = os.path.dirname(os.path.abspath(__file__))

    sa_sections = []
    ma_sections = []

    for mode in modes:
        # num_agents = 1
        try:
            prompts_1 = external.get_external_transition(
                prompt=original_prompt,
                agent_completions=[main_only_code],
                num_agents=1,
                mode=mode,
                expert_model=args.expert_model if mode == "expert_edits" else None,
            )
            # get_external_transition returns [main_prompt] for single-agent
            if isinstance(prompts_1, (list, tuple)):
                main_prompt_1 = prompts_1[-1]
            else:
                main_prompt_1 = str(prompts_1)

            sa_sections.append(
                "\n".join(
                    [
                        f"=== MODE: {mode} | num_agents=1 ===",
                        "",
                        "MAIN PROMPT:",
                        main_prompt_1,
                        "",
                    ]
                )
            )
        except Exception as e:
            msg = f"[agents=1] Error in mode '{mode}': {e}"
            print(msg)
            sa_sections.append(f"=== MODE: {mode} | num_agents=1 ===\nERROR: {e}\n")

        # num_agents = 2
        try:
            prompts_2 = external.get_external_transition(
                prompt=original_prompt,
                agent_completions=(aux_code, main_code),
                num_agents=2,
                mode=mode,
                expert_model=args.expert_model if mode == "expert_edits" else None,
            )
            # get_external_transition returns (aux_prompt, main_prompt) for two agents
            if isinstance(prompts_2, (list, tuple)) and len(prompts_2) == 2:
                aux_prompt_2, main_prompt_2 = prompts_2
            else:
                # Fallback to consistent formatting
                aux_prompt_2 = ""
                main_prompt_2 = (
                    prompts_2[0]
                    if isinstance(prompts_2, (list, tuple))
                    else str(prompts_2)
                )

            ma_sections.append(
                "\n".join(
                    [
                        f"=== MODE: {mode} | num_agents=2 ===",
                        "",
                        "AUX PROMPT:",
                        aux_prompt_2,
                        "",
                        "MAIN PROMPT:",
                        main_prompt_2,
                        "",
                    ]
                )
            )
        except Exception as e:
            msg = f"[agents=2] Error in mode '{mode}': {e}"
            print(msg)
            ma_sections.append(f"=== MODE: {mode} | num_agents=2 ===\nERROR: {e}\n")

    # Write combined outputs
    sa_path = os.path.join(out_dir, "prompts_sa.txt")
    with open(sa_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sa_sections).rstrip() + "\n")
    print(f"Wrote {sa_path}")

    ma_path = os.path.join(out_dir, "prompts_ma.txt")
    with open(ma_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ma_sections).rstrip() + "\n")
    print(f"Wrote {ma_path}")


if __name__ == "__main__":
    main()
