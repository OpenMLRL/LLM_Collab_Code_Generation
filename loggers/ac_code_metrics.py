from collections import defaultdict
from typing import Any, Dict, List

from loggers.mt_code_logger import (
    aggregate_mt_humaneval_metrics_for_logging,
    mt_humaneval_logger,
)


def build_ac_code_metrics_callback(num_agents: int, num_turns: int):
    def callback(rollouts: List[Any]) -> Dict[str, float]:
        return aggregate_ac_code_metrics(
            rollouts,
            num_agents=max(1, int(num_agents)),
            num_turns=max(1, int(num_turns)),
        )

    return callback


def aggregate_ac_code_metrics(
    rollouts: List[Any], *, num_agents: int, num_turns: int
) -> Dict[str, float]:
    if not rollouts:
        return {}

    first_item = _first_batch_item(rollouts)
    if not first_item:
        return {}

    entry_point = str(first_item.get("entry_point") or "")
    test_case = str(first_item.get("test") or "")
    prompt = str(first_item.get("prompt") or "")
    if not entry_point or not test_case:
        return {}

    by_generation: Dict[int, Dict[int, Dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    max_generation = 0
    max_turn = 0

    for sample in rollouts:
        metadata = getattr(sample, "metadata", {}) or {}
        agent_idx = int(getattr(sample, "agent_idx", 0))
        generation_idx = int(metadata.get("generation_idx", 0))
        turn_idx = int(metadata.get("turn_idx", 0))
        max_generation = max(max_generation, generation_idx)
        max_turn = max(max_turn, turn_idx)
        by_generation[generation_idx][turn_idx][agent_idx] = str(
            getattr(sample, "completion", "") or ""
        )

    sample_count = max_generation + 1
    turns = max(num_turns, max_turn + 1)
    agent_completions_turns: List[List[List[str]]] = []

    for agent_idx in range(num_agents):
        per_sample: List[List[str]] = []
        for generation_idx in range(sample_count):
            turns_for_sample: List[str] = []
            for turn_idx in range(turns):
                turns_for_sample.append(
                    by_generation[generation_idx][turn_idx].get(agent_idx, "")
                )
            per_sample.append(turns_for_sample)
        agent_completions_turns.append(per_sample)

    detailed = mt_humaneval_logger(
        agent_completions_turns=agent_completions_turns,
        test_cases=[test_case] * sample_count,
        entry_points=[entry_point] * sample_count,
        prompts=[prompt] * sample_count,
    )
    return aggregate_mt_humaneval_metrics_for_logging(detailed, num_turns=turns)


def _first_batch_item(rollouts: List[Any]) -> Dict[str, Any]:
    for sample in rollouts:
        metadata = getattr(sample, "metadata", {}) or {}
        item = metadata.get("batch_item")
        if isinstance(item, dict):
            return item
    return {}
