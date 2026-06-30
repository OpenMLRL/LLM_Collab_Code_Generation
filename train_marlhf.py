"""Run iterative MARLHF with pairwise preference reward reconstruction."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config, add_config_args, make_run_id, parse_overrides
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent

RL_SCRIPTS = {
    "grpo": "train_grpo.py",
    "magrpo": "train_magrpo.py",
    "maac": "train_maac.py",
    "iac": "train_iac.py",
}


def _as_override(key: str, value: Any) -> str:
    return f"{key}={repr(value) if isinstance(value, list) else value}"


def _run(script: str, config_path: Path, overrides: List[str]) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / script),
        "--config",
        str(config_path),
    ]
    if overrides:
        cmd.extend(["--override", *overrides])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _section(config: Config, name: str) -> Dict[str, Any]:
    return config.get_section(name) if hasattr(config, "get_section") else {}


def _algorithm(config: Config) -> str:
    algorithm = str(config.get("marlhf.algorithm", "magrpo")).strip().lower()
    if algorithm not in RL_SCRIPTS:
        supported = ", ".join(sorted(RL_SCRIPTS))
        raise ValueError(f"marlhf.algorithm must be one of: {supported}.")
    return algorithm


def _num_agents(config: Config, algorithm: str) -> int:
    if algorithm == "grpo":
        return 1
    return int(_section(config, algorithm).get("num_agents", 2))


def _run_name(config: Config, algorithm: str) -> str:
    wandb_section = _section(config, "wandb")
    return (
        wandb_section.get("name")
        or wandb_section.get("run_name")
        or f"marlhf_{algorithm}_{config.get('dataset.type', 'code')}"
    )


def _wandb_id(algorithm: str, run_id: str) -> str:
    raw = f"marlhf_{algorithm}_{run_id}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def _rl_steps_per_iteration(config: Config, algorithm: str, train_items: int) -> int:
    if algorithm not in RL_SCRIPTS:
        return 0
    section = _section(config, algorithm)
    return (
        int(train_items)
        * int(section.get("num_train_epochs", 1))
        * int(section.get("num_generations", 1))
        * int(section.get("num_turns", 1))
    )


def _saved_agent_paths(policy_dir: Path, num_agents: int) -> List[str]:
    if num_agents == 1 and not (policy_dir / "agent_0").exists():
        return [str(policy_dir)]
    return [str(policy_dir / f"agent_{idx}") for idx in range(num_agents)]


def _agent_overrides(agent_paths: Optional[List[str]]) -> List[str]:
    if not agent_paths:
        return []
    return [
        _as_override("agents", agent_paths),
        _as_override("preference.policy_agents", agent_paths),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train iterative MARLHF.")
    add_config_args(parser)
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Please provide --config.")

    config_path = Path(args.config).resolve()
    config = Config(str(config_path))
    if args.override:
        config.update(parse_overrides(args.override))

    algorithm = _algorithm(config)
    num_iterations = int(config.get("marlhf.num_iterations", 1))
    if num_iterations < 1:
        raise ValueError("marlhf.num_iterations must be >= 1.")

    num_agents = _num_agents(config, algorithm)
    run_id = make_run_id()
    output_base = Path(config.get("output.base_dir", "output_marlhf"))
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base
    run_dir = output_base / f"marlhf_{algorithm}_job_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_config_path = run_dir / "config.yaml"
    if hasattr(config, "save"):
        config.save(str(effective_config_path))
    else:
        effective_config_path = config_path
    train_items = len(
        load_dataset(config.get("dataset.name"), split=config.get("dataset.train_split"))
    )
    rl_steps_per_iteration = _rl_steps_per_iteration(config, algorithm, train_items)
    wandb_name = _run_name(config, algorithm)
    wandb_run_id = _wandb_id(algorithm, run_id)

    current_agents: Optional[List[str]] = None
    if config.get("agents") is not None:
        agents = config.get("agents")
        if not isinstance(agents, list):
            raise ValueError("agents must be a list when set.")
        current_agents = [str(path) for path in agents]

    for iteration in range(num_iterations):
        iter_dir = run_dir / f"iteration_{iteration + 1:03d}"
        pref_path = iter_dir / "preferences.jsonl"
        reward_dir = iter_dir / "reward_model"
        policy_dir = iter_dir / "policy"
        iter_dir.mkdir(parents=True, exist_ok=True)

        common_agent_overrides = _agent_overrides(current_agents)
        collect_overrides = [
            *common_agent_overrides,
            _as_override("preference.num_agents", num_agents),
            _as_override("preference.output_path", str(pref_path)),
            _as_override("preference.overwrite", True),
        ]
        _run("preferences/collect_preferences.py", effective_config_path, collect_overrides)

        reward_overrides = [
            _as_override("reward_model.buffer_path", str(pref_path)),
            _as_override("reward_model.output_dir", str(reward_dir)),
        ]
        _run("preferences/train_reward_model.py", effective_config_path, reward_overrides)

        train_overrides = [
            *common_agent_overrides,
            _as_override("reward.type", "model"),
            _as_override("reward_model.path", str(reward_dir)),
            _as_override("reward_processor.shift", 0),
            _as_override("output.save_final_model", True),
            _as_override("output.save_path", str(policy_dir)),
            _as_override("wandb.name", wandb_name),
            _as_override("wandb.id", wandb_run_id),
            _as_override("wandb.resume", "allow" if iteration == 0 else "must"),
        ]
        if algorithm in RL_SCRIPTS:
            train_overrides.extend(
                [
                    _as_override(
                        f"{algorithm}.initial_env_step",
                        iteration * rl_steps_per_iteration,
                    ),
                    _as_override(f"{algorithm}.iteration_index", iteration + 1),
                    _as_override(f"{algorithm}.iteration_total", num_iterations),
                ]
            )
        if algorithm in {"grpo", "maac", "iac"}:
            train_overrides.append(_as_override(f"{algorithm}.reward_shift", 0))
        print(
            "MARLHF iteration "
            f"{iteration + 1}/{num_iterations}: "
            f"wandb_id={wandb_run_id} "
            f"resume={'allow' if iteration == 0 else 'must'} "
            f"initial_env_step={iteration * rl_steps_per_iteration}",
            flush=True,
        )
        _run(RL_SCRIPTS[algorithm], effective_config_path, train_overrides)
        current_agents = _saved_agent_paths(policy_dir, num_agents)


if __name__ == "__main__":
    main()
