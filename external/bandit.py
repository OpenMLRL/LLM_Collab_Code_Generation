from typing import Tuple

from .common import build_first_turn_prompts


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
    original_prompt_flag: bool = True,
    previous_response_flag: bool = False,
    num_agent: int = 2,
) -> Tuple[str, str]:
    """
    Bandit mode: make follow-up prompts identical to the canonical first-turn
    prompts. No analysis and no "Revise ..." instructions. Ignores completions.

    Returns (aux_prompt, main_prompt). For single-agent, aux will be an empty string
    and the caller should use only the main prompt.
    """
    # Build the canonical first-turn prompts (context + instructions)
    aux_base, main_base = build_first_turn_prompts(original_prompt, entry_point)
    if int(num_agent) == 1:
        # For single-agent, aux prompt is unused; return an empty aux and main_base
        return "", main_base
    return aux_base, main_base

