from typing import Any, Dict, List, Optional

import numpy as np


def mt_humaneval_logger(
    agent_completions_turns: List[List[List[str]]],
    test_cases: List[str],
    entry_points: List[str],
    prompts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-turn logger for code generation tasks (HumanEval/CoopHumanEval) with aux + main function collaboration.

    Args:
        agent_completions_turns: List per agent -> per sample -> per turn completions
        test_cases: List of test cases
        entry_points: List of entry point function names
        prompts: Optional list of prompts for import extraction

    Returns:
        List of metric dictionaries with multi-turn information
    """
    from loggers.code_logger import code_reward_logger

    all_metrics = []

    # Derive completions for aux/main from agent_completions_turns
    # agent_completions_turns shape: [num_agents][num_samples][num_turns]
    if not agent_completions_turns or len(agent_completions_turns) == 0:
        return []

    num_agents = len(agent_completions_turns)
    # Use first agent as aux (if available), last agent as main
    aux_per_sample_turns = (
        agent_completions_turns[0] if num_agents >= 2 else [[""] * len(agent_completions_turns[0][0]) for _ in range(len(agent_completions_turns[0]))]
    )
    main_per_sample_turns = agent_completions_turns[-1]

    # Get number of turns from the data (assume consistent across samples)
    num_turns = len(main_per_sample_turns[0]) if main_per_sample_turns and main_per_sample_turns[0] else 0

    for i in range(len(test_cases)):
        sample_metrics = {
            "sample_id": i,
            "entry_point": entry_points[i],
        }

        # Process each turn without early termination
        turn_rewards = []

        for turn_idx in range(num_turns):
            # Extract completions for this turn
            turn_completions1 = [
                (
                    aux_per_sample_turns[j][turn_idx]
                    if j < len(aux_per_sample_turns)
                    and turn_idx < len(aux_per_sample_turns[j])
                    else ""
                )
                for j in range(len(test_cases))
            ]
            turn_completions2 = [
                (
                    main_per_sample_turns[j][turn_idx]
                    if j < len(main_per_sample_turns)
                    and turn_idx < len(main_per_sample_turns[j])
                    else ""
                )
                for j in range(len(test_cases))
            ]

            # Get metrics for this turn using the single-turn logger
            turn_metrics = code_reward_logger(
                [turn_completions1[i]],
                [turn_completions2[i]],
                [test_cases[i]],
                [entry_points[i]],
                [prompts[i]] if prompts else None,
            )

            if turn_metrics:
                turn_metric = turn_metrics[0]
                turn_rewards.append(turn_metric["total_reward"])

                # Add turn-specific metrics
                for key, value in turn_metric.items():
                    if key not in ["sample_id", "entry_point"]:
                        sample_metrics[f"turn_{turn_idx + 1}/{key}"] = value

            # sample_metrics["num_turns"] = num_turns  # optional if needed elsewhere

        # No turn-to-turn improvement metrics

        # Overall metrics
        if turn_rewards:
            sample_metrics["overall/best_turn_reward"] = max(turn_rewards)
            sample_metrics["overall/final_turn_reward"] = turn_rewards[-1]
            sample_metrics["overall/avg_turn_reward"] = np.mean(turn_rewards)

        all_metrics.append(sample_metrics)

    return all_metrics


def aggregate_mt_humaneval_metrics_for_logging(
    metrics_list: List[Dict[str, Any]], num_turns: int = 2
) -> Dict[str, float]:
    """
    Aggregate multi-turn metrics from multiple samples for wandb logging.

    Args:
        metrics_list: List of sample metrics from mt_humaneval_logger
        num_turns: Number of turns to aggregate metrics for

    Returns:
        Dictionary of aggregated metrics
    """
    if not metrics_list:
        return {}

    aggregated: Dict[str, float] = {}

    # No overall early termination or num_turns aggregation

    # Turn-specific metrics
    for turn in range(1, num_turns + 1):
        turn_prefix = f"turn_{turn}"

        def _mean(raw_key: str):
            key = f"{turn_prefix}/{raw_key}"
            values = [sample[key] for sample in metrics_list if key in sample]
            if not values:
                return None
            return float(np.mean(values))

        syntax_rewards = [
            sample[f"{turn_prefix}/level_2_reward"]
            for sample in metrics_list
            if f"{turn_prefix}/level_2_reward" in sample
        ]
        if syntax_rewards:
            aggregated[f"{turn_prefix}/code/syntax_valid_rate"] = float(
                np.mean([1.0 if value > 0 else 0.0 for value in syntax_rewards])
            )

        metric_map = {
            "definition_reward": "level_1_reward",
            "syntax_reward": "level_2_reward",
            "execution_reward": "test_reward",
            "tests_passed": "passed_tests",
            "tests_total": "total_tests",
            "test_pass_rate": "passed_rate",
            "timeouts": "timeout_num",
            "bonus_reward": "bonus_reward",
            "collab_aux_usage": "aux_usage_bonus",
            "collab_non_wrapper": "anti_wrapper_bonus",
            "collab_ignored_aux_call_penalty": "called_wo_used_deduction",
            "total_reward": "total_reward",
            "gated_total_reward": "gated_total_reward",
        }

        for clean_key, raw_key in metric_map.items():
            value = _mean(raw_key)
            if value is not None:
                aggregated[f"{turn_prefix}/code/{clean_key}"] = value

        # No improvement metrics

    # Only return per-turn aggregates
    return aggregated
