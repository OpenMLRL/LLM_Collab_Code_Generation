from typing import Any, Callable, Dict, List, Tuple, Union, Optional

# Mode implementations live alongside this file
from . import expert_edits
from . import level_feedback
from . import level_passed
from . import passed
from . import plain

# -----------------------------
# Context resolver API
# -----------------------------
_context_resolver: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None


def set_context_resolver(fn: Callable[[str], Optional[Dict[str, Any]]]):
    """Register a resolver that maps prompt -> context dict.

    Expected dict keys:
      - entry_point: str
      - tests_eval: str
      - tests_sandbox: Optional[str]
    """
    global _context_resolver
    _context_resolver = fn


def get_context(prompt: str) -> Optional[Dict[str, Any]]:
    """Resolve context for a given prompt using the registered resolver."""
    if _context_resolver is None:
        return None
    try:
        return _context_resolver(prompt)
    except Exception:
        return None


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
        agent_completions: Selected completions from previous turn (one per agent).
        num_agents: Number of agents (1 or 2).
        mode: External transition mode name (default: "expert_edits").
        **kwargs: Mode-specific parameters (e.g., expert_model, retries).

    Returns:
        A list/tuple of full prompts for each agent to use in the next turn.
    """
    if int(num_agents) not in (1, 2):
        raise ValueError(
            f"External transition supports 1 or 2 agents, got {num_agents}."
        )

    if not isinstance(agent_completions, (list, tuple)) or len(
        agent_completions
    ) != int(num_agents):
        raise ValueError(
            f"Expected {num_agents} agent completions but got {len(agent_completions) if isinstance(agent_completions, (list, tuple)) else 'invalid type'}"
        )

    # Route to the requested mode implementation
    mode = (mode or "").lower()
    # Pull common flags controlling prompt composition
    original_prompt_flag = kwargs.get("original_prompt", False)
    previous_response_flag = kwargs.get("previous_response", True)

    if mode == "expert_edits":
        if int(num_agents) == 1:
            main_comp = agent_completions[0]
            aux_comp = ""  # isolate aux in single-agent mode
        else:
            aux_comp, main_comp = agent_completions[0], agent_completions[1]
        original_prompt, aux_edits, main_edits = expert_edits.add_expert_edits(
            prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            expert_model=kwargs.get("expert_model", "deepseek-coder"),
            max_retries=kwargs.get("max_retries", 3),
            num_agents=int(num_agents),
        )

        # Format the follow-up prompts for each agent using this mode's formatter
        ctx = get_context(prompt) or {}
        entry_point = ctx.get("entry_point", "")
        aux_prompt, main_prompt = expert_edits.format_followup_prompts(
            original_prompt=original_prompt,
            aux_edits=aux_edits,
            main_edits=main_edits,
            entry_point=entry_point,
            aux_completion=aux_comp,
            main_completion=main_comp,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            num_agent=int(num_agents),
        )

        # Print preview
        print("\n" + "=" * 60)
        print("EXTERNAL MODE PREVIEW: expert_edits")
        print("-" * 60)
        if int(num_agents) > 1:
            print("AUX PROMPT:\n" + aux_prompt)
            print("-" * 60)
        print("MAIN PROMPT:\n" + main_prompt)
        print("=" * 60 + "\n")

        return (aux_prompt, main_prompt) if int(num_agents) > 1 else [main_prompt]

    if mode == "level_feedback":
        if int(num_agents) == 1:
            main_comp = agent_completions[0]
            aux_comp = ""
        else:
            aux_comp, main_comp = agent_completions[0], agent_completions[1]
        ctx = get_context(prompt) or {}
        entry_point = ctx.get("entry_point", "")
        test_code = ctx.get("tests_sandbox") or ctx.get("tests_eval", "")
        aux_prompt, main_prompt = level_feedback.format_followup_prompts(
            original_prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            test_code=test_code,
            entry_point=entry_point,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            num_agent=int(num_agents),
        )
        print("\n" + "=" * 60)
        print("EXTERNAL MODE PREVIEW: level_feedback")
        print("-" * 60)
        if int(num_agents) > 1:
            print("AUX PROMPT:\n" + aux_prompt)
            print("-" * 60)
        print("MAIN PROMPT:\n" + main_prompt)
        print("=" * 60 + "\n")
        return (aux_prompt, main_prompt) if int(num_agents) > 1 else [main_prompt]

    if mode == "level_passed":
        if int(num_agents) == 1:
            main_comp = agent_completions[0]
            aux_comp = ""
        else:
            aux_comp, main_comp = agent_completions[0], agent_completions[1]
        ctx = get_context(prompt) or {}
        entry_point = ctx.get("entry_point", "")
        test_code = ctx.get("tests_sandbox") or ctx.get("tests_eval", "")
        aux_prompt, main_prompt = level_passed.format_followup_prompts(
            original_prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            test_code=test_code,
            entry_point=entry_point,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            num_agent=int(num_agents),
        )
        print("\n" + "=" * 60)
        print("EXTERNAL MODE PREVIEW: level_passed")
        print("-" * 60)
        if int(num_agents) > 1:
            print("AUX PROMPT:\n" + aux_prompt)
            print("-" * 60)
        print("MAIN PROMPT:\n" + main_prompt)
        print("=" * 60 + "\n")
        return (aux_prompt, main_prompt) if int(num_agents) > 1 else [main_prompt]

    if mode == "passed":
        if int(num_agents) == 1:
            main_comp = agent_completions[0]
            aux_comp = ""
        else:
            aux_comp, main_comp = agent_completions[0], agent_completions[1]
        ctx = get_context(prompt) or {}
        entry_point = ctx.get("entry_point", "")
        test_code = ctx.get("tests_sandbox") or ctx.get("tests_eval", "")
        aux_prompt, main_prompt = passed.format_followup_prompts(
            original_prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            test_code=test_code,
            entry_point=entry_point,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            num_agent=int(num_agents),
        )
        print("\n" + "=" * 60)
        print("EXTERNAL MODE PREVIEW: passed")
        print("-" * 60)
        if int(num_agents) > 1:
            print("AUX PROMPT:\n" + aux_prompt)
            print("-" * 60)
        print("MAIN PROMPT:\n" + main_prompt)
        print("=" * 60 + "\n")
        return (aux_prompt, main_prompt) if int(num_agents) > 1 else [main_prompt]

    if mode == "plain":
        if int(num_agents) == 1:
            main_comp = agent_completions[0]
            aux_comp = ""
        else:
            aux_comp, main_comp = agent_completions[0], agent_completions[1]
        ctx = get_context(prompt) or {}
        entry_point = ctx.get("entry_point", "")
        test_code = ctx.get("tests_sandbox") or ctx.get("tests_eval", "")
        aux_prompt, main_prompt = plain.format_followup_prompts(
            original_prompt=prompt,
            aux_completion=aux_comp,
            main_completion=main_comp,
            test_code=test_code,
            entry_point=entry_point,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            num_agent=int(num_agents),
        )
        print("\n" + "=" * 60)
        print("EXTERNAL MODE PREVIEW: plain")
        print("-" * 60)
        if int(num_agents) > 1:
            print("AUX PROMPT:\n" + aux_prompt)
            print("-" * 60)
        print("MAIN PROMPT:\n" + main_prompt)
        print("=" * 60 + "\n")
        return (aux_prompt, main_prompt) if int(num_agents) > 1 else [main_prompt]

    supported = ["expert_edits", "level_feedback", "level_passed", "passed", "plain"]
    raise NotImplementedError(
        f"External transition mode '{mode}' is not implemented yet. Supported: {', '.join(supported)}"
    )
