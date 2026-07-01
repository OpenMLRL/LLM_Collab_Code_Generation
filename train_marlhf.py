import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.reinforce import MARLHFConfig, MARLHFTrainer
from preference_train_common import run_preference_training


def main():
    parser = argparse.ArgumentParser(
        description="Train MARLHF with configurable code-generation datasets"
    )
    add_config_args(parser)
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Please provide a configuration file using --config")

    config = Config(args.config)
    if args.override:
        config.update(parse_overrides(args.override))

    run_preference_training(
        config=config,
        section_name="marlhf",
        args_cls=MARLHFConfig,
        trainer_cls=MARLHFTrainer,
        algorithm_name="MARLHF",
    )


if __name__ == "__main__":
    main()
