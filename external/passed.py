from typing import Tuple

from .level_feedback import analyze_code


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
) -> Tuple[str, str]:
    """
    Minimal pass/fail signal mode based on sandbox tests.
    Returns prompts that include the previous code and a single pass/fail signal.
    """
    r = analyze_code(original_prompt, aux_completion, main_completion, test_code, entry_point)

    # Consider 'passed' only if syntax is OK, main is defined, and all tests pass (when present)
    tests_total = r.get("tests_total", 0)
    tests_passed = r.get("tests_passed", 0)
    syntax_ok = r.get("syntax_ok", False)
    main_defined = r.get("main_defined", False)

    passed_all = syntax_ok and main_defined and (tests_total > 0 and tests_passed == tests_total)
    signal = "All levels passed" if passed_all else "Not all levels passed"

    aux_lines = [
        "Your previous aux(...) implementation:",
        r.get("aux_func") or "<no implementation found>",
        "",
        f"Signal: {signal}",
        "",
        "Revise your aux(...) if needed. Output ONLY the function code.",
    ]

    main_lines = [
        "Your previous main implementation:",
        r.get("main_func") or "<no implementation found>",
        "",
        f"Signal: {signal}",
        "",
        f"Revise your {entry_point or 'main'}(...) if needed. Output ONLY the function code.",
    ]

    return ("\n".join(aux_lines), "\n".join(main_lines))

