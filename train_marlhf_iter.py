import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.preference import MARLHFIterConfig, MARLHFIterTrainer
from preference_train_common import run_preference_training


def main():
    parser = argparse.ArgumentParser(
        description="Train iterative MARLHF with configurable code-generation datasets"
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
        section_name="marlhf_iter",
        args_cls=MARLHFIterConfig,
        trainer_cls=MARLHFIterTrainer,
        algorithm_name="MARLHFIter",
    )


if __name__ == "__main__":
    main()
