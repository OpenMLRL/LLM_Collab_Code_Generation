import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "marlhf_iter_che_config.yaml"
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
for path in (COMLRL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Config, add_config_args, parse_overrides
from comlrl.trainers.preference import MARLHFIterConfig, MARLHFIterTrainer
from preference_train_common import run_preference_training


def main():
    parser = argparse.ArgumentParser(
        description="Train iterative MARLHF with configurable code-generation datasets"
    )
    add_config_args(parser)
    args = parser.parse_args()
    config = Config(args.config or str(DEFAULT_CONFIG))
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
