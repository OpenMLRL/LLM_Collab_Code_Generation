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
    Produce concise level_passed prompts for each agent using previous code + signals.
    """
    r = analyze_code(original_prompt, aux_completion, main_completion, test_code, entry_point)

    # Build compact signals
    # Implementation signals
    aux_impl = "OK" if r["aux_defined"] else "No implementation found in required structure"
    main_impl = "OK" if r["main_defined"] else "No implementation found in required structure"

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

    # Aux usage signals (for main)
    if not r["main_defined"]:
        aux_use_sig = "Main missing; cannot check aux usage"
    else:
        if not r["aux_called"]:
            aux_use_sig = "Main does not call aux"
        elif r["aux_called_but_not_used"]:
            aux_use_sig = "Aux called but result not used properly"
        else:
            aux_use_sig = "Aux call present and used"

    # Compose prompts
    aux_lines = [
        "Your previous aux(...) implementation:",
        r.get("aux_func") or "<no implementation found>",
        "",
        "Signals:",
        f"- Implementation: {aux_impl}",
        f"- Syntax: {syntax_sig}",
        f"- Tests: {test_sig}",
        "",
        "Revise your aux(...) accordingly. Output ONLY the function code.",
    ]

    main_lines = [
        "Your previous main implementation:",
        r.get("main_func") or "<no implementation found>",
        "",
        "Signals:",
        f"- Implementation: {main_impl}",
        f"- Syntax: {syntax_sig}",
        f"- Tests: {test_sig}",
        f"- Aux usage: {aux_use_sig}",
        "",
        f"Revise your {entry_point}(...) accordingly. Output ONLY the function code.",
    ]

    return ("\n".join(aux_lines), "\n".join(main_lines))

