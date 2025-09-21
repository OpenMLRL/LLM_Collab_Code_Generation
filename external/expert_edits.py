import json
import os
import re
from typing import List, Tuple

from anthropic import Anthropic
from openai import OpenAI
from rewards.code_utils import (
    concatenate_functions,
    extract_imports_from_prompt,
    extract_specific_function,
)
from .common import build_first_turn_prompts


def _extract_last_json_from_response(response_text: str) -> dict:
    """Extract the last valid JSON object from a text response."""
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    potential_jsons = re.findall(json_pattern, response_text, re.DOTALL)
    for json_str in reversed(potential_jsons):
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in response")


def add_expert_edits(
    prompt: str,
    aux_completion: str,
    main_completion: str,
    expert_model: str = "claude-3-5-sonnet-20241022",
    max_retries: int = 3,
    num_agents: int = 2,
) -> Tuple[str, str, str]:
    """
    Produce expert edits for previous turn outputs.

    Returns a tuple of (original_prompt, aux_edits, main_edits), where edits are
    strings that can be injected into next-turn prompts by the mode formatter.

    This refactors previous get_expert_feedback() behavior into a more general
    "edits provider" that returns the original prompt alongside per-agent edits.
    """

    imports = extract_imports_from_prompt(prompt)
    if int(num_agents) == 1:
        combined_code = concatenate_functions("", main_completion, imports)
        expert_prompt = f"""You are an expert reviewing code written by a single agent for the task: {prompt}

Your job is to return proposed edits for the main function as JSON with the key 'main'.
Guidelines:
1) If the target function is missing, provide a minimal correct implementation as the edit.
2) If it exists, provide concise edits that would help pass the provided unit tests.
3) If no changes are needed, respond with short note: "Perfect! No changes needed!" for that field.

Provide only JSON in the following format: {{ "main": <string with edits or replacement> }}
Here is the current code to consider:
{combined_code}
"""
    else:
        combined_code = concatenate_functions(aux_completion, main_completion, imports)
        expert_prompt = f"""You are an expert reviewing code collaboratively written by two agents for the task: {prompt}
The auxiliary agent implements a helper function (aux), and the main agent implements the task entry function.

Your job is to return proposed edits for each agent as JSON with keys 'aux' and 'main'.
Guidelines:
1) If a function is missing, provide a minimal correct implementation as the edit.
2) If both exist, provide concise edits that would help pass the provided unit tests.
3) If no changes are needed, respond with short note: "Perfect! No changes needed!" for that field.

Provide only JSON in the following format: {{ "aux": <string with edits or replacement>, "main": <string with edits or replacement> }}
Here is the current combined code to consider:
{combined_code}
"""

    last_error = None
    for _ in range(max_retries):
        try:
            response_text = None
            if "claude" in expert_model.lower():
                client = Anthropic()
                resp = client.messages.create(
                    model=expert_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": expert_prompt}],
                )
                response_text = resp.content[0].text
            elif "deepseek" in expert_model.lower():
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
                model_name = (
                    "deepseek-coder"
                    if expert_model == "deepseek-coder"
                    else expert_model
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,
                )
                response_text = resp.choices[0].message.content
            elif "qwen3-coder" in expert_model.lower():
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                )
                model_name = (
                    "qwen3-coder" if expert_model == "qwen3-coder" else expert_model
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,
                )
                response_text = resp.choices[0].message.content
            else:
                raise ValueError(f"Unsupported expert model: {expert_model}")

            parsed = _extract_last_json_from_response(response_text)
            aux_edits = str(parsed.get("aux", ""))
            main_edits = str(parsed.get("main", ""))

            print("\n" + "=" * 60)
            print("EXPERT EDITS")
            print("\n--- FULL EXPERT RESPONSE ---")
            print(response_text)
            print("\n--- EXTRACTED EDITS ---")
            print("AUX EDITS:")
            print(aux_edits)
            print("\nMAIN EDITS:")
            print(main_edits)
            print("=" * 60 + "\n")

            return prompt, aux_edits, main_edits
        except Exception as e:
            last_error = e

    # Fallback: no edits; return original prompt with empty edits
    print(f"Expert edits failed, using original prompt. Error: {last_error}")
    return prompt, "", ""


def _extract_function_params_from_prompt(prompt_text: str) -> List[str]:
    import re

    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def format_followup_prompts(
    original_prompt: str,
    aux_edits: str,
    main_edits: str,
    entry_point: str = "",
    aux_completion: str = "",
    main_completion: str = "",
    original_prompt_flag: bool = False,
    previous_response_flag: bool = True,
    num_agent: int = 2,
) -> Tuple[str, str]:
    """
    Format the 2+ turn prompts for expert_edits mode to match other modes:
    - Include the previous implementation snippet
    - Include an expert-edited snippet line
    - Ask to revise the function and output only code
    """

    target_entry = entry_point or "main"

    # Single-agent: only build main prompt; no aux references
    if int(num_agent) == 1:
        main_lines: List[str] = []

        prev_main = (
            extract_specific_function(main_completion or "", target_entry)
            or "<no implementation found>"
        )

        if original_prompt_flag:
            _aux_base, main_base = build_first_turn_prompts(
                original_prompt, target_entry
            )
            main_lines.extend([main_base, ""])  # context then blank line

        if previous_response_flag:
            main_lines.extend(
                [
                    "Your previous implementation:",
                    prev_main,
                    "",
                ]
            )

        main_lines.extend(
            [
                "Here is an edited snippet from an expert model:",
                (main_edits or "").strip(),
                "",
                f"Revise your {target_entry}(...) accordingly. Output ONLY the function code with no extra text.",
            ]
        )

        return ("", "\n".join(main_lines))

    # Multi-agent (default): keep original aux + main collaboration logic
    prev_aux = (
        extract_specific_function(aux_completion or "", "aux")
        or "<no implementation found>"
    )
    prev_main = (
        extract_specific_function(main_completion or "", target_entry)
        or "<no implementation found>"
    )

    aux_lines: List[str] = []
    main_lines: List[str] = []

    if original_prompt_flag:
        aux_base, main_base = build_first_turn_prompts(original_prompt, target_entry)
        aux_lines.extend([aux_base, ""])  # add a blank line after context
        main_lines.extend([main_base, ""])

    if previous_response_flag:
        aux_lines.extend(["Your previous aux(...) implementation:", prev_aux, ""])
        main_lines.extend(["Your previous main implementation:", prev_main, ""])

    aux_lines.extend(
        [
            "Here is edited snippet from an expert model:",
            (aux_edits or "").strip(),
            "",
            "Revise your aux(...) accordingly. Output ONLY the function code with no extra text.",
        ]
    )

    main_lines.extend(
        [
            "Here is edited snippet from an expert model:",
            (main_edits or "").strip(),
            "",
            f"Revise your {target_entry}(...) accordingly. Output ONLY the function code with no extra text.",
        ]
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
