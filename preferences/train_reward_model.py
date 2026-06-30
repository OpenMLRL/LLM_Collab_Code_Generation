import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config, add_config_args, parse_overrides
from rewards.code_rewards import format_reward_model_input


def _torch_dtype(name: str | None):
    return getattr(torch, name) if name else None


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample_text(record: Dict[str, Any], sample_id: int) -> str:
    sample = record["joint_samples"][sample_id]
    return format_reward_model_input(
        record.get("prompt", ""),
        sample.get("aux", ""),
        sample.get("main", ""),
    )


def _pair_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for record in records:
        pref = record.get("pair_preference")
        if pref is None:
            continue
        examples.append(
            {
                "chosen": _sample_text(record, int(pref["chosen"])),
                "rejected": _sample_text(record, int(pref["rejected"])),
            }
        )
    return examples


def _reward_values(model, tokenizer, texts: List[str], device: torch.device, max_length: int):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return model(**inputs).logits.view(-1)


def _iter_batches(examples: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pairwise preference reward model.")
    add_config_args(parser)
    args = parser.parse_args()

    if not args.config:
        raise ValueError("Please provide --config.")
    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    seed = int(config.get("reward_model.seed", config.get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)

    buffer_path = config.get("reward_model.buffer_path", config.get("preference.output_path"))
    if not buffer_path:
        raise ValueError("reward_model.buffer_path or preference.output_path must be set.")
    records = _load_records(Path(buffer_path))

    examples = _pair_examples(records)
    if not examples:
        raise ValueError("No pair preference examples were found.")

    base_model = config.get("reward_model.base_model", config.get("agent_model.name"))
    output_dir = Path(
        config.get(
            "reward_model.output_dir",
            f"output_reward_model/{config.get('dataset.type')}_pair",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        str(config.get("reward_model.device", "cuda" if torch.cuda.is_available() else "cpu"))
    )
    max_length = int(config.get("reward_model.max_length", 2048))
    batch_size = int(config.get("reward_model.train_batch_size", 2))
    epochs = int(config.get("reward_model.num_train_epochs", 1))
    learning_rate = float(config.get("reward_model.learning_rate", 1e-5))
    dtype = _torch_dtype(
        config.get("reward_model.torch_dtype", config.get("agent_model.torch_dtype"))
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"num_labels": 1}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, **model_kwargs
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        random.shuffle(examples)
        losses = []
        for batch in _iter_batches(examples, batch_size):
            optimizer.zero_grad()
            chosen_values = _reward_values(
                model,
                tokenizer,
                [item["chosen"] for item in batch],
                device,
                max_length,
            )
            rejected_values = _reward_values(
                model,
                tokenizer,
                [item["rejected"] for item in batch],
                device,
                max_length,
            )
            loss = -F.logsigmoid(chosen_values - rejected_values).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = sum(losses) / len(losses)
        print(f"epoch={epoch + 1} loss={mean_loss:.6f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": base_model,
        "buffer_path": str(buffer_path),
        "num_examples": len(examples),
        "loss": "bradley_terry_pair",
    }
    (output_dir / "reward_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps({"reward_model_path": str(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
