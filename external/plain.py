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
    Plain mode: no signals or diagnostics. Just include previous code and
    instructions to revise. Mirrors the structure used by other modes.
    """
    r = analyze_code(
        original_prompt,
        aux_completion,
        main_completion,
        test_code,
        entry_point,
        num_agent=num_agent,
    )

    # Single-agent: only main prompt with no aux references
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
                f"Revise your {entry_point or 'main'}(...) accordingly. Output ONLY the function code.",
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
            "Revise your aux(...) accordingly. Output ONLY the function code.",
        ]
    )

    main_lines.extend(
        [
            f"Revise your {entry_point or 'main'}(...) accordingly. Output ONLY the function code.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
