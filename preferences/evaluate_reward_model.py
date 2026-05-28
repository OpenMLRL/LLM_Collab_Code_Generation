import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from config import Config, add_config_args, parse_overrides
from rewards.code_rewards import BradleyTerryRewardFunction


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reward model scores to oracle rewards.")
    add_config_args(parser)
    args = parser.parse_args()

    if not args.config:
        raise ValueError("Please provide --config.")
    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    buffer_path = config.get("reward_model.buffer_path", config.get("preference.output_path"))
    if not buffer_path:
        raise ValueError("reward_model.buffer_path or preference.output_path must be set.")

    model_path = config.get("reward_model.path", config.get("reward_model.output_dir"))
    if not model_path:
        raise ValueError("reward_model.path or reward_model.output_dir must be set.")
    config.update({"reward_model": {"path": model_path}})

    output_dir = Path(
        config.get("reward_model.eval_output_dir", str(Path(model_path) / "eval"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = BradleyTerryRewardFunction.from_config(config)
    records = _load_records(Path(buffer_path))
    rows = []
    sample_index = 0
    for record in records:
        samples = record["joint_samples"]
        aux_outputs = [sample.get("aux", "") for sample in samples]
        main_outputs = [sample.get("main", "") for sample in samples]
        scores = scorer(
            aux_outputs,
            main_outputs,
            batch_items=[record] * len(samples),
        )
        for sample, model_reward in zip(samples, scores):
            oracle_reward = float(sample["oracle_reward"])
            rows.append(
                {
                    "index": sample_index,
                    "task_id": record.get("task_id", ""),
                    "sample_id": int(sample["sample_id"]),
                    "oracle_reward": oracle_reward,
                    "model_reward": float(model_reward),
                    "diff": float(model_reward) - oracle_reward,
                }
            )
            sample_index += 1

    jsonl_path = output_dir / "reward_diffs.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = output_dir / "reward_diffs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [row["index"] for row in rows]
    y = [row["diff"] for row in rows]
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, linewidth=1.5)
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("sample index")
    plt.ylabel("tilde reward - oracle reward")
    plt.title("Reward model error")
    plt.tight_layout()
    plot_path = output_dir / "reward_diff_curve.png"
    plt.savefig(plot_path, dpi=200)

    mean_diff = sum(y) / len(y)
    mean_abs_diff = sum(abs(v) for v in y) / len(y)
    summary = {
        "num_samples": len(rows),
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    summary_path = output_dir / "reward_diff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
