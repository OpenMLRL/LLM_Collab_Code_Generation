from typing import Tuple


def build_first_turn_prompts(original_prompt: str, entry_point: str) -> Tuple[str, str]:
    """
    Build the canonical first-turn prompts used by all modes.
    Returns (aux_prompt, main_prompt) strings.
    """
    params_str = "..."
    # Keep generic since params may not be reliably parsed here.

    aux_fmt = (
        "Original problem and instructions:\n\n"
        "Create a helper function for this coding problem.\n\n"
        f"Problem:\n{original_prompt}\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Output ONLY the function code, no explanations or examples\n"
        "- Do NOT include markdown code blocks (```python)\n"
        "- Do NOT include any text before or after the function\n"
        "- Do NOT include test cases or example usage\n"
        "- Create a helper function named 'aux' that can assist the main function\n"
        "- The function should return useful data for solving the problem\n\n"
        "Your output should follow this format:\n\n"
        "def aux(...):\n # your function code here\nreturn result\n"
    )

    main_fmt = (
        "Original problem and instructions:\n\n"
        "Solve this coding problem by implementing the required function.\n\n"
        f"Problem:\n{original_prompt}\n\n"
        "You have access to a helper function: aux(...)\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Output ONLY the function code, no explanations or examples\n"
        "- Do NOT include markdown code blocks (```python)\n"
        "- Do NOT include any text before or after the function\n"
        "- Do NOT include test cases or example usage\n"
        "- Do NOT redefine the aux() function\n"
        + (
            f"- Implement ONLY the '{entry_point}' function as specified\n"
            if entry_point
            else "- Implement ONLY the required function as specified\n"
        )
        + "- You can call aux() to assign value to a variable within your function if helpful\n\n"
        + "Your output should follow this format:\n\n"
        + (
            f"def {entry_point}({params_str}):\n # your function code here\nreturn result\n"
            if entry_point
            else "def <entry_point>(...):\n # your function code here\nreturn result\n"
        )
    )

    return aux_fmt, main_fmt
