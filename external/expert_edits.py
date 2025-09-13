import json
import os
import re
from typing import Tuple

from anthropic import Anthropic
from openai import OpenAI
from rewards.code_utils import concatenate_functions, extract_imports_from_prompt


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
) -> Tuple[str, str, str]:
    """
    Produce expert edits for previous turn outputs.

    Returns a tuple of (original_prompt, aux_edits, main_edits), where edits are
    strings that can be injected into next-turn prompts by the mode formatter.

    This refactors previous get_expert_feedback() behavior into a more general
    "edits provider" that returns the original prompt alongside per-agent edits.
    """

    imports = extract_imports_from_prompt(prompt)
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
                model_name = "deepseek-coder" if expert_model == "deepseek-coder" else expert_model
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
                model_name = "qwen3-coder" if expert_model == "qwen3-coder" else expert_model
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


def format_followup_prompts(
    original_prompt: str,
    aux_edits: str,
    main_edits: str,
) -> Tuple[str, str]:
    """
    Mode-specific formatter for next-turn prompts.

    Current behavior (mirrors the existing setup):
    - If edits are provided, append a brief note with the edits to the original prompt.
    - Otherwise, return the original prompt unchanged.
    """
    def _append_edits(base: str, edits: str) -> str:
        return (
            f"{base}\n\nHere is the feedback from an expert:\n{edits}" if edits.strip() else base
        )

    aux_prompt = _append_edits(original_prompt, aux_edits)
    main_prompt = _append_edits(original_prompt, main_edits)
    return aux_prompt, main_prompt

