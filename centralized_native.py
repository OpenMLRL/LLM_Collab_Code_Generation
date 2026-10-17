"""Native centralized CoopHE prompt: one programmer writes aux and main.

The concatenated adapter embeds the Auxiliary and Main agent prompts, so the
problem appears twice and each copy demands "only the function", which clashes
with the two tagged sections. This prompt states the problem once and merges
both roles' instructions; the unchanged parser splits the two functions back
into the agents' outputs for the same reward.
"""

import re
from typing import Any, Dict, Sequence

from centralized_comparator import CoopHECentralizedComparatorAdapter


def _has_function_params(prompt: str) -> bool:
    # Same rule as the role formatters' extract_function_params_from_prompt.
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt)
    return bool(match and any(p.strip() for p in match.group(1).split(",")))


def build_native_centralized_prompt(example: Dict[str, Any]) -> str:
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    if not _has_function_params(prompt) or not entry_point:
        # The role formatters return this message for the same rows.
        return "Error: Could not extract function information from prompt."
    signature = CoopHECentralizedComparatorAdapter._main_signature(example)
    return f"""Solve this coding problem by writing two functions yourself: a helper function
named 'aux' and the required main function '{entry_point}'. The main function can
call aux(). A downstream evaluator combines both functions and runs the tests.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Write the helper function 'aux' first. It should return useful data for solving
  the problem.
- Then implement ONLY the '{entry_point}' function as specified. You can call aux()
  to assign a value to a variable within it if helpful. Do NOT redefine aux() inside it.
- Output only code: no explanations, examples, test cases, or markdown code blocks.
- Do not copy placeholders such as ..., result, or pass.
- Use the exact section tags below so the two functions can be separated.

<auxiliary>
def aux(...):
    # your auxiliary code here
    return result
</auxiliary>

<main>
{signature}:
    # your main code here
    return result
</main>
"""


class CoopHENativeCentralizedAdapter(CoopHECentralizedComparatorAdapter):
    """State the problem once; parse the same Auxiliary/Main sections."""

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        if len(agent_prompts) != 2:
            raise ValueError("CoopHE centralized generation requires exactly 2 agents.")
        return build_native_centralized_prompt(batch_item)
