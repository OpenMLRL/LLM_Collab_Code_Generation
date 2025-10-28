from typing import Tuple, List, Optional

from .level_feedback import analyze_code
from .common import build_first_turn_prompts


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
    *,
    previous_prompts: bool = False,
    previous_responses: bool = True,
    memory_mode: str = "last",
    num_agent: int = 2,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
) -> Tuple[str, str]:
    """
    Minimal pass/fail signal mode based on sandbox tests.
    Returns prompts that include the previous code and a single pass/fail signal.
    """
    r = analyze_code(
        original_prompt,
        aux_completion,
        main_completion,
        test_code,
        entry_point,
        num_agent=num_agent,
    )

    # Consider 'passed' only if syntax is OK, main is defined, and all tests pass (when present)
    tests_total = r.get("tests_total", 0)
    tests_passed = r.get("tests_passed", 0)
    syntax_ok = r.get("syntax_ok", False)
    main_defined = r.get("main_defined", False)

    passed_all = (
        syntax_ok and main_defined and (tests_total > 0 and tests_passed == tests_total)
    )
    signal = "All levels passed" if passed_all else "Not all levels passed"

    # Normalize histories
    memory_mode = (memory_mode or "last").lower()
    # If full history has only one prior turn, render identically to 'last'
    if memory_mode == "full":
        try:
            counts_p = [len(x or []) for x in (prompt_history_per_agent or [])]
            counts_r = [len(x or []) for x in (response_history_per_agent or [])]
            if counts_p and max(counts_p) <= 1 and counts_r and max(counts_r) <= 1:
                memory_mode = "last"
        except Exception:
            pass
    if prompt_history_per_agent is None:
        prompt_history_per_agent = [[] for _ in range(int(num_agent))]
    if response_history_per_agent is None:
        response_history_per_agent = [[] for _ in range(int(num_agent))]

    # Single-agent: only main prompt
    if int(num_agent) == 1:
        main_lines = []
        if memory_mode == "full":
            if previous_prompts and prompt_history_per_agent and prompt_history_per_agent[0]:
                main_lines.extend(["History: previous prompts:"])
                for t, ph in enumerate(prompt_history_per_agent[0], start=1):
                    main_lines.append(f"- Turn {t} prompt:\n{ph}")
                main_lines.append("")
            if previous_responses and response_history_per_agent and response_history_per_agent[0]:
                main_lines.extend(["History: your previous responses:"])
                for t, resp in enumerate(response_history_per_agent[0], start=1):
                    main_lines.append(f"- Turn {t} response:\n{resp}")
                main_lines.append("")
        elif memory_mode == "last":
            if previous_prompts:
                _aux_base, main_base = build_first_turn_prompts(
                    original_prompt, entry_point
                )
                main_lines.extend([main_base, ""])  # context then blank line
            if previous_responses:
                main_lines.extend(
                    [
                        "Your previous implementation:",
                        r.get("main_func") or "<no implementation found>",
                        "",
                    ]
                )
        elif memory_mode == "memoryful":
            pass
        main_lines.extend(
            [
                f"Signal: {signal}",
                "",
                f"Revise your {entry_point or 'main'}(...) if needed. Output ONLY the function code.",
            ]
        )
        return ("", "\n".join(main_lines))

    aux_lines = []
    main_lines = []

    if memory_mode == "full":
        if previous_prompts:
            if prompt_history_per_agent and len(prompt_history_per_agent) >= 2:
                aux_ph = prompt_history_per_agent[0]
                main_ph = prompt_history_per_agent[1]
                if aux_ph:
                    aux_lines.append("History: previous prompts:")
                    for t, ph in enumerate(aux_ph, start=1):
                        aux_lines.append(f"- Turn {t} prompt:\n{ph}")
                    aux_lines.append("")
                if main_ph:
                    main_lines.append("History: previous prompts:")
                    for t, ph in enumerate(main_ph, start=1):
                        main_lines.append(f"- Turn {t} prompt:\n{ph}")
                    main_lines.append("")
        if previous_responses:
            if response_history_per_agent and len(response_history_per_agent) >= 2:
                aux_rh = response_history_per_agent[0]
                main_rh = response_history_per_agent[1]
                if aux_rh:
                    aux_lines.append("History: your previous aux(...) responses:")
                    for t, resp in enumerate(aux_rh, start=1):
                        aux_lines.append(f"- Turn {t} response:\n{resp}")
                    aux_lines.append("")
                if main_rh:
                    main_lines.append("History: your previous main responses:")
                    for t, resp in enumerate(main_rh, start=1):
                        main_lines.append(f"- Turn {t} response:\n{resp}")
                    main_lines.append("")
    elif memory_mode == "last":
        if previous_prompts:
            aux_base, main_base = build_first_turn_prompts(original_prompt, entry_point)
            aux_lines.extend([aux_base, ""])  # context then blank line
            main_lines.extend([main_base, ""])  # context then blank line
        if previous_responses:
            aux_lines.extend(
                [
                    "Your previous aux(...) implementation:",
                    r.get("aux_func") or "<no implementation found>",
                    "",
                ]
            )
            main_lines.extend(
                [
                    "Your previous main implementation:",
                    r.get("main_func") or "<no implementation found>",
                    "",
                ]
            )
    elif memory_mode == "memoryful":
        pass

    aux_lines.extend(
        [
            f"Signal: {signal}",
            "",
            "Revise your aux(...) if needed. Output ONLY the function code.",
        ]
    )

    main_lines.extend(
        [
            f"Signal: {signal}",
            "",
            f"Revise your {entry_point or 'main'}(...) if needed. Output ONLY the function code.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
