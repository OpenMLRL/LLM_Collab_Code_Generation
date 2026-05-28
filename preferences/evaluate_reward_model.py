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


FIELDNAMES = [
    "metric",
    "index",
    "value",
    "task_id",
    "sample_id",
    "oracle_reward",
    "model_reward",
    "chosen",
    "rejected",
    "chosen_oracle_reward",
    "rejected_oracle_reward",
    "chosen_model_reward",
    "rejected_model_reward",
    "oracle_gap",
]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _empty_row(metric: str, index: int, value: float, task_id: str) -> Dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "metric": metric,
            "index": index,
            "value": value,
            "task_id": task_id,
        }
    )
    return row


def _plot_metric(ax, rows: List[Dict[str, Any]], metric: str, ylabel: str) -> None:
    values = [float(row["value"]) for row in rows if row["metric"] == metric]
    if not values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(metric)
        return

    ax.plot(range(len(values)), values, linewidth=1.2)
    ax.set_xlabel("index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} mean={sum(values) / len(values):.4f}")
    if metric == "reward_value_error":
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    else:
        ax.set_ylim(-0.05, 1.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reward model value and preference errors.")
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
    rows: List[Dict[str, Any]] = []
    indices = {
        "reward_value_error": 0,
        "pair_error": 0,
        "ranking_error": 0,
    }

    for record in records:
        task_id = str(record.get("task_id", ""))
        samples = record["joint_samples"]
        aux_outputs = [sample.get("aux", "") for sample in samples]
        main_outputs = [sample.get("main", "") for sample in samples]
        scores = scorer(
            aux_outputs,
            main_outputs,
            batch_items=[record] * len(samples),
        )
        by_id = {
            int(sample["sample_id"]): {
                "oracle": float(sample["oracle_reward"]),
                "model": float(score),
            }
            for sample, score in zip(samples, scores)
        }

        for sample in samples:
            sample_id = int(sample["sample_id"])
            oracle_reward = by_id[sample_id]["oracle"]
            model_reward = by_id[sample_id]["model"]
            row = _empty_row(
                "reward_value_error",
                indices["reward_value_error"],
                model_reward - oracle_reward,
                task_id,
            )
            row.update(
                {
                    "sample_id": sample_id,
                    "oracle_reward": oracle_reward,
                    "model_reward": model_reward,
                }
            )
            rows.append(row)
            indices["reward_value_error"] += 1

        for pref in record["pair_preferences"]:
            chosen = int(pref["chosen"])
            rejected = int(pref["rejected"])
            error = float(by_id[chosen]["model"] <= by_id[rejected]["model"])
            row = _empty_row("pair_error", indices["pair_error"], error, task_id)
            row.update(
                {
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_oracle_reward": by_id[chosen]["oracle"],
                    "rejected_oracle_reward": by_id[rejected]["oracle"],
                    "chosen_model_reward": by_id[chosen]["model"],
                    "rejected_model_reward": by_id[rejected]["model"],
                    "oracle_gap": by_id[chosen]["oracle"] - by_id[rejected]["oracle"],
                }
            )
            rows.append(row)
            indices["pair_error"] += 1

        ranking = [
            (int(item["sample_id"]), float(item["score"]))
            for item in record["ranking_preferences"]
        ]
        for better_idx in range(len(ranking)):
            for worse_idx in range(better_idx + 1, len(ranking)):
                better, better_oracle = ranking[better_idx]
                worse, worse_oracle = ranking[worse_idx]
                if better_oracle == worse_oracle:
                    continue
                error = float(by_id[better]["model"] <= by_id[worse]["model"])
                row = _empty_row(
                    "ranking_error",
                    indices["ranking_error"],
                    error,
                    task_id,
                )
                row.update(
                    {
                        "chosen": better,
                        "rejected": worse,
                        "chosen_oracle_reward": by_id[better]["oracle"],
                        "rejected_oracle_reward": by_id[worse]["oracle"],
                        "chosen_model_reward": by_id[better]["model"],
                        "rejected_model_reward": by_id[worse]["model"],
                        "oracle_gap": by_id[better]["oracle"] - by_id[worse]["oracle"],
                    }
                )
                rows.append(row)
                indices["ranking_error"] += 1

    csv_path = output_dir / "reward_model_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    _plot_metric(axes[0], rows, "reward_value_error", "tilde reward - oracle")
    _plot_metric(axes[1], rows, "pair_error", "0/1 error")
    _plot_metric(axes[2], rows, "ranking_error", "0/1 error")
    fig.tight_layout()
    plot_path = output_dir / "reward_model_eval.png"
    fig.savefig(plot_path, dpi=200)

    metrics = {}
    for metric in indices:
        values = [float(row["value"]) for row in rows if row["metric"] == metric]
        if values:
            metrics[metric] = sum(values) / len(values)
    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "plot_path": str(plot_path),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
