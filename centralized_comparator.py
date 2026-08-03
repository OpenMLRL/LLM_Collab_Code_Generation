"""Centralized comparator prompting and parsing for CoopHE code generation."""

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


class CoopHECentralizedComparatorAdapter:
    """Preserve the Auxiliary/Main centralized protocol used by CoopHE."""

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        if len(agent_prompts) != 2:
            raise ValueError("CoopHE centralized generation requires exactly 2 agents.")
        entry_signature = self._main_signature(batch_item)
        return f"""You are acting as one centralized coordinator for two code-generation agents.

Your job is to produce the exact outputs that two decentralized agents would submit:
one Auxiliary output and one Main output. The two outputs must work together.

Auxiliary agent original prompt:
{agent_prompts[0]}

Main agent original prompt:
{agent_prompts[1]}

IMPORTANT INSTRUCTIONS:
- Return exactly two complete code snippets: one for Auxiliary and one for Main.
- The Auxiliary snippet must define the helper function aux(...).
- The Main snippet must define only the required main function from the problem.
- Do not copy placeholders such as required_function, ..., result, or pass.
- Do not include explanations, examples, tests, markdown fences, or extra text.
- Use the exact section tags below so the parser can split the two outputs.

<auxiliary>
def aux(...):
    # your auxiliary code here
    return result
</auxiliary>

<main>
{entry_signature}:
    # your main code here
    return result
</main>
"""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        if num_agents != 2:
            raise ValueError("CoopHE centralized generation requires exactly 2 agents.")
        auxiliary, main = self._parse_completion(completion, batch_item=batch_item)
        return [auxiliary, main]

    @staticmethod
    def _main_signature(batch_item: Dict[str, Any]) -> str:
        entry_point = str(batch_item.get("entry_point") or "").strip()
        prompt = str(batch_item.get("prompt") or "")
        if entry_point:
            pattern = rf"def\s+{re.escape(entry_point)}\s*\(([^)]*)\)"
            match = re.search(pattern, prompt)
            if match:
                return f"def {entry_point}({match.group(1).strip()})"
            return f"def {entry_point}(...)"
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", prompt)
        if match:
            return f"def {match.group(1)}({match.group(2).strip()})"
        return "def main(...)"

    def _parse_completion(
        self,
        text: str,
        *,
        batch_item: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        partial_auxiliary: Optional[str] = None
        partial_main: Optional[str] = None

        def remember(
            auxiliary: Optional[str],
            main: Optional[str],
        ) -> Optional[Tuple[str, str]]:
            nonlocal partial_auxiliary, partial_main
            clean_auxiliary = (
                self._clean_section(auxiliary) if auxiliary is not None else None
            )
            clean_main = self._clean_section(main) if main is not None else None
            if clean_auxiliary is not None and partial_auxiliary is None:
                partial_auxiliary = clean_auxiliary
            if clean_main is not None and partial_main is None:
                partial_main = clean_main
            if clean_auxiliary is not None and clean_main is not None:
                return clean_auxiliary, clean_main
            return None

        result = remember(
            self._extract_tagged_section(text, "auxiliary"),
            self._extract_tagged_section(text, "main"),
        )
        if result is not None:
            return result

        result = remember(*self._extract_labeled_sections(text))
        if result is not None:
            return result

        result = remember(*self._extract_json_sections(text))
        if result is not None:
            return result

        result = remember(*self._extract_fenced_code_sections(text))
        if result is not None:
            return result

        result = remember(*self._extract_function_sections(text, batch_item=batch_item))
        if result is not None:
            return result

        if partial_auxiliary is not None or partial_main is not None:
            return partial_auxiliary or "", partial_main or ""
        return "", ""

    @staticmethod
    def _extract_tagged_section(text: str, tag: str) -> Optional[str]:
        pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def _extract_labeled_sections(
        text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        auxiliary_label = re.search(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:auxiliary|aux)\s*(?:code|output)?\s*:\s*$",
            text,
        )
        main_label = re.search(
            r"(?im)^\s*(?:#{1,6}\s*)?main\s*(?:code|output)?\s*:\s*$",
            text,
        )
        if auxiliary_label is None or main_label is None:
            if auxiliary_label is not None:
                return text[auxiliary_label.end() :], None
            if main_label is not None:
                return None, text[main_label.end() :]
            return None, None
        if auxiliary_label.start() < main_label.start():
            return (
                text[auxiliary_label.end() : main_label.start()],
                text[main_label.end() :],
            )
        return (
            text[auxiliary_label.end() :],
            text[main_label.end() : auxiliary_label.start()],
        )

    @staticmethod
    def _extract_json_sections(
        text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        value = text.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL
        )
        if fence_match:
            value = fence_match.group(1).strip()

        candidates = [value]
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            candidates.append(value[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            auxiliary = parsed.get("auxiliary", parsed.get("aux"))
            main = parsed.get("main")
            if auxiliary is not None or main is not None:
                return (
                    str(auxiliary) if auxiliary is not None else None,
                    str(main) if main is not None else None,
                )
        return None, None

    @staticmethod
    def _extract_fenced_code_sections(
        text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        blocks = re.findall(
            r"```(?:python)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if len(blocks) >= 2:
            return blocks[0], blocks[1]
        return None, None

    @classmethod
    def _extract_function_sections(
        cls,
        text: str,
        *,
        batch_item: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        entry_point = None
        if isinstance(batch_item, dict):
            value = batch_item.get("entry_point")
            if value is not None:
                entry_point = str(value).strip() or None

        auxiliary = cls._extract_python_function(text, "aux")
        main = cls._extract_python_function(text, entry_point) if entry_point else None
        if main is None:
            for name in cls._python_function_names(text):
                if name != "aux":
                    main = cls._extract_python_function(text, name)
                    break
        return auxiliary, main

    @staticmethod
    def _python_function_names(text: str) -> List[str]:
        return [
            match.group(1)
            for match in re.finditer(
                r"(?m)^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text
            )
        ]

    @staticmethod
    def _extract_python_function(
        text: str,
        function_name: Optional[str],
    ) -> Optional[str]:
        if not function_name:
            return None
        pattern = re.compile(rf"(?m)^([ \t]*)def\s+{re.escape(function_name)}\s*\(")
        match = pattern.search(text)
        if match is None:
            return None

        lines = text[match.start() :].splitlines()
        if not lines:
            return None
        base_indent = len(lines[0]) - len(lines[0].lstrip(" \t"))
        selected = [lines[0]]
        for line in lines[1:]:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" \t"))
            if stripped and indent <= base_indent:
                if re.match(r"(def|class)\s+", stripped):
                    break
                if re.match(r"<?/?(?:auxiliary|main)>?:?$", stripped, re.I):
                    break
                if re.match(
                    r"(?:auxiliary|aux|main)\s*(?:code|output)?\s*:",
                    stripped,
                    re.I,
                ):
                    break
            selected.append(line)
        return "\n".join(selected).strip()

    @staticmethod
    def _clean_section(text: str) -> str:
        value = text.strip()
        fence_match = re.fullmatch(
            r"```(?:python)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL
        )
        return fence_match.group(1).strip() if fence_match else value
