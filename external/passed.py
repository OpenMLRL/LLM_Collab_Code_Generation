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

    # Single-agent: only main prompt
    if int(num_agent) == 1:
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
                f"Signal: {signal}",
                "",
                f"Revise your {entry_point or 'main'}(...) if needed. Output ONLY the function code.",
            ]
        )
        return ("", "\n".join(main_lines))

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
