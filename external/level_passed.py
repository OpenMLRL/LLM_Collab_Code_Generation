from typing import Tuple

from .level_feedback import analyze_code
from .common import build_first_turn_prompts


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
    original_prompt_flag: bool = False,
    previous_response_flag: bool = True,
    num_agent: int = 2,
) -> Tuple[str, str]:
    """
    Produce concise level_passed prompts for each agent using previous code + signals.
    """
    r = analyze_code(
        original_prompt,
        aux_completion,
        main_completion,
        test_code,
        entry_point,
        num_agent=num_agent,
    )

    # Build compact signals
    # Syntax signals
    syntax_sig = "Syntax correct" if r["syntax_ok"] else "Syntax not correct"

    # Test signals
    if r["tests_total"] == 0:
        test_sig = "No tests found"
    elif r["tests_passed"] == r["tests_total"]:
        test_sig = "Passed all tests"
    elif r["tests_passed"] == 0:
        test_sig = "No tests passed"
    else:
        test_sig = f"Passed {r['tests_passed']}/{r['tests_total']} tests"

    # Single-agent: only main prompt; exclude aux logic/signals
    if int(num_agent) == 1:
        main_impl = (
            "OK"
            if r["main_defined"]
            else "No implementation found in required structure"
        )
        main_lines = []
        if original_prompt_flag:
            _aux_base, main_base = build_first_turn_prompts(
                original_prompt, entry_point
            )
            main_lines.extend([main_base, ""])  # context then blank line
        if previous_response_flag:
            main_lines.extend(
                [
                    "Your previous implementation:",
                    r.get("main_func") or "<no implementation found>",
                    "",
                ]
            )
        main_lines.extend(
            [
                "Signals:",
                f"- Implementation: {main_impl}",
                f"- Syntax: {syntax_sig}",
                f"- Tests: {test_sig}",
                "",
                f"Revise your {entry_point}(...) accordingly. Output ONLY the function code.",
            ]
        )
        return ("", "\n".join(main_lines))

    # Multi-agent (default): original aux + main collaboration
    aux_impl = (
        "OK" if r["aux_defined"] else "No implementation found in required structure"
    )
    main_impl = (
        "OK" if r["main_defined"] else "No implementation found in required structure"
    )

    if not r["main_defined"]:
        aux_use_sig = "Main missing; cannot check aux usage"
    else:
        if not r["aux_called"]:
            aux_use_sig = "Main does not call aux"
        elif r["aux_called_but_not_used"]:
            aux_use_sig = "Aux called but result not used properly"
        else:
            aux_use_sig = "Aux call present and used"

    aux_lines = []
    main_lines = []

    if original_prompt_flag:
        aux_base, main_base = build_first_turn_prompts(original_prompt, entry_point)
        aux_lines.extend([aux_base, ""])  # context then blank line
        main_lines.extend([main_base, ""])  # context then blank line

    if previous_response_flag:
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

    aux_lines.extend(
        [
            "Signals:",
            f"- Implementation: {aux_impl}",
            f"- Syntax: {syntax_sig}",
            f"- Tests: {test_sig}",
            "",
            "Revise your aux(...) accordingly. Output ONLY the function code.",
        ]
    )

    main_lines.extend(
        [
            "Signals:",
            f"- Implementation: {main_impl}",
            f"- Syntax: {syntax_sig}",
            f"- Tests: {test_sig}",
            f"- Aux usage: {aux_use_sig}",
            "",
            f"Revise your {entry_point}(...) accordingly. Output ONLY the function code.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
