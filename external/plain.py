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
    Plain mode: no signals or diagnostics. Just include previous code and
    instructions to revise. Mirrors the structure used by other modes.
    """
    r = analyze_code(original_prompt, aux_completion, main_completion, test_code, entry_point)

    aux_lines = [
        "Your previous aux(...) implementation:",
        r.get("aux_func") or "<no implementation found>",
        "",
        "Revise your aux(...) accordingly. Output ONLY the function code.",
    ]

    main_lines = [
        "Your previous main implementation:",
        r.get("main_func") or "<no implementation found>",
        "",
        f"Revise your {entry_point or 'main'}(...) accordingly. Output ONLY the function code.",
    ]

    return ("\n".join(aux_lines), "\n".join(main_lines))

