from typing import List, Tuple, Union

# Mode implementations live alongside this file
from . import expert_edits


def get_external_transition(
    prompt: str,
    agent_completions: Union[List[str], Tuple[str, str]],
    num_agents: int = 2,
    mode: str = "expert_edits",
    **kwargs,
) -> Union[List[str], Tuple[str, str]]:
    """
    Wrapper for external transition modes.

    For turn > 1, returns the full next-turn prompts for each agent based on
    the selected `mode`. For the first turn, the trainer should not call this
    (returns are intended for subsequent turns only).

    Args:
        prompt: Original problem prompt.
        agent_completions: Best completions from previous turn (one per agent).
        num_agents: Number of agents supported (currently 2).
        mode: External transition mode name (default: "expert_edits").
        **kwargs: Mode-specific parameters (e.g., expert_model, retries).

    Returns:
        A list/tuple of full prompts for each agent to use in the next turn.
    """
    if num_agents != 2:
        raise ValueError(
            f"External transition currently supports 2 agents, got {num_agents}."
        )

    if not isinstance(agent_completions, (list, tuple)) or len(agent_completions) != 2:
        raise ValueError(
            f"Expected 2 agent completions but got {len(agent_completions) if isinstance(agent_completions, (list, tuple)) else 'invalid type'}"
        )

    # Route to the requested mode implementation
    mode = (mode or "").lower()
    if mode in ("expert_edits", "expert", "edits"):
        aux_comp, main_comp = agent_completions[0], agent_completions[1]
        original_prompt, aux_edits, main_edits = expert_edits.add_expert_edits(
            prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            expert_model=kwargs.get("expert_model", "deepseek-coder"),
            max_retries=kwargs.get("max_retries", 3),
        )

        # Format the follow-up prompts for each agent using this mode's formatter
        aux_prompt, main_prompt = expert_edits.format_followup_prompts(
            original_prompt=original_prompt,
            aux_edits=aux_edits,
            main_edits=main_edits,
        )
        return (aux_prompt, main_prompt)

    raise ValueError(f"Unsupported external transition mode: {mode}")

