"""Checkpoint launch contracts, without model downloads or real training."""

import builtins
from collections import UserDict
from copy import deepcopy
import importlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

import checkpoint_launch as checkpoint
from config import Config


ROOT = Path(__file__).resolve().parents[1]
SAVED_IDENTITY = {"id": "saved-run", "project": "saved-project", "entity": "saved-team"}
BASE_WANDB = {
    "project": "configured-project",
    "entity": "configured-team",
    "name": "experiment-name",
    "tags": ["code", "resume"],
    "dir": "logs",
    "config_sections": {"dataset": {"name": "local-dataset"}},
}


def forbid_metadata_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "comlrl.utils" and "read_checkpoint_metadata" in fromlist:
            pytest.fail("checkpoint metadata API was imported unnecessarily")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_helper_import_and_default_options_have_no_checkpoint_io(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    forbid_metadata_import(monkeypatch)
    namespace = runpy.run_path(str(Path(checkpoint.__file__)))
    for config in ({}, {"checkpoint": None}, {"checkpoint": {}}):
        assert namespace["parse_checkpoint_options"](config) == {}
    for options in (
        {},
        {"resume_from_checkpoint": None},
        {"checkpoint_dir": str(tmp_path / "periodic-only"), "checkpoint_interval": 3},
    ):
        result = namespace["prepare_wandb_config"](
            BASE_WANDB, section={}, checkpoint_options=options
        )
        assert result == BASE_WANDB
        assert result is not BASE_WANDB
    assert not list(tmp_path.iterdir())


def test_parse_preserves_names_null_paths_and_mapping_without_touching_paths(tmp_path):
    options = UserDict(
        resume_from_checkpoint=str(tmp_path / "missing-source"),
        checkpoint_dir=str(tmp_path / "missing-output"),
        checkpoint_interval=3,
        checkpoint_total_limit=2,
    )
    parsed = checkpoint.parse_checkpoint_options({"checkpoint": options})
    assert parsed == options
    assert parsed is not options
    assert not list(tmp_path.iterdir())
    assert checkpoint.parse_checkpoint_options(
        {
            "checkpoint": {
                "resume_from_checkpoint": None,
                "checkpoint_dir": None,
                "checkpoint_total_limit": None,
            }
        }
    ) == {
        "resume_from_checkpoint": None,
        "checkpoint_dir": None,
        "checkpoint_total_limit": None,
    }


@pytest.mark.parametrize("section", [False, True, [], "checkpoint-dir", 1])
def test_parse_rejects_non_mapping_sections(section):
    with pytest.raises(ValueError, match="checkpoint"):
        checkpoint.parse_checkpoint_options({"checkpoint": section})


@pytest.mark.parametrize("key", ["resume", "save_steps", "checkpoint_intervall"])
def test_parse_rejects_unknown_option_names(key):
    with pytest.raises(ValueError, match=key):
        checkpoint.parse_checkpoint_options({"checkpoint": {key: 1}})


@pytest.mark.parametrize("key", ["resume_from_checkpoint", "checkpoint_dir"])
@pytest.mark.parametrize("value", ["", "   ", 1, False, [], Path("checkpoint")])
def test_parse_rejects_non_string_or_empty_paths(key, value):
    with pytest.raises(ValueError, match=key):
        checkpoint.parse_checkpoint_options({"checkpoint": {key: value}})


@pytest.mark.parametrize("key", ["checkpoint_interval", "checkpoint_total_limit"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "2"])
def test_parse_requires_positive_integer_counts(key, value):
    with pytest.raises(ValueError, match=key):
        checkpoint.parse_checkpoint_options({"checkpoint": {key: value}})


def test_parse_rejects_null_interval():
    with pytest.raises(ValueError, match="checkpoint_interval"):
        checkpoint.parse_checkpoint_options(
            {"checkpoint": {"checkpoint_interval": None}}
        )


@pytest.mark.parametrize(
    "section", [None, False, {"enabled": False}, {"mode": "disabled"}]
)
def test_disabled_wandb_never_imports_metadata(section, monkeypatch):
    forbid_metadata_import(monkeypatch)
    assert (
        checkpoint.prepare_wandb_config(
            BASE_WANDB,
            section=section,
            checkpoint_options={"resume_from_checkpoint": "missing-checkpoint"},
        )
        is None
    )


def test_absent_prepared_wandb_never_imports_metadata(monkeypatch):
    forbid_metadata_import(monkeypatch)
    assert (
        checkpoint.prepare_wandb_config(
            None,
            section={},
            checkpoint_options={"resume_from_checkpoint": "missing-checkpoint"},
        )
        is None
    )


@pytest.mark.parametrize(
    "section",
    [
        {"id": "explicit-run"},
        {"resume": "must"},
        {"id": None},
        {"resume": None},
        {"id": "explicit-run", "resume": "never"},
    ],
)
def test_explicit_identity_keys_skip_metadata_even_when_null(section, monkeypatch):
    forbid_metadata_import(monkeypatch)
    original = deepcopy(BASE_WANDB)
    result = checkpoint.prepare_wandb_config(
        BASE_WANDB,
        section=section,
        checkpoint_options={"resume_from_checkpoint": "missing-checkpoint"},
    )
    assert result == {**BASE_WANDB, **section}
    assert result is not BASE_WANDB
    assert BASE_WANDB == original


@pytest.mark.parametrize("identity", [{"id": "prepared-run"}, {"resume": None}])
def test_prepared_identity_also_skips_metadata(identity, monkeypatch):
    forbid_metadata_import(monkeypatch)
    prepared = {**BASE_WANDB, **identity}
    assert (
        checkpoint.prepare_wandb_config(
            prepared,
            section={},
            checkpoint_options={"resume_from_checkpoint": "missing-checkpoint"},
        )
        == prepared
    )


@pytest.mark.parametrize("metadata", [{"wandb": SAVED_IDENTITY}, {"wandb": None}, {}])
def test_metadata_restores_only_saved_identity_and_keeps_launch_settings(
    metadata, monkeypatch
):
    import comlrl.utils

    calls = []

    def read_metadata(path):
        calls.append(path)
        return deepcopy(metadata)

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", read_metadata)
    original = deepcopy(BASE_WANDB)
    result = checkpoint.prepare_wandb_config(
        BASE_WANDB,
        section={"project": "configured-project"},
        checkpoint_options={"resume_from_checkpoint": "checkpoint/state.pt"},
    )
    assert calls == ["checkpoint/state.pt"]
    expected = dict(BASE_WANDB)
    if metadata.get("wandb"):
        expected.update(SAVED_IDENTITY, resume="allow")
    assert result == expected
    assert result is not BASE_WANDB
    assert BASE_WANDB == original


LAUNCHERS = [
    ("train_ac", "ac", "IACTrainer"),
    ("train_grpo", "grpo", "MAGRPOTrainer"),
    ("train_iac", "iac", "IACTrainer"),
    ("train_maac", "maac", "MAACTrainer"),
    ("train_magrpo", "magrpo", "MAGRPOTrainer"),
    ("preference_train_common", "madpo", "MADPOConfig"),
    ("preference_train_common", "marlhf", "MARLHFConfig"),
    ("preference_train_common", "madpo_iter", "MADPOIterConfig"),
    ("preference_train_common", "marlhf_iter", "MARLHFIterConfig"),
]


class LocalDataset(list):
    def select(self, indices):
        return LocalDataset(self[index] for index in indices)


@pytest.fixture(params=LAUNCHERS, ids=[item[1] for item in LAUNCHERS])
def launch(request, monkeypatch, tmp_path):
    module_name, section, trainer_or_config = request.param
    module = importlib.import_module(module_name)
    config = Config(str(ROOT / "configs" / f"{section}_che_config.yaml"))
    config.update(
        {
            section: {"num_turns": 1, "agent_devices": "cpu"},
            "output": {
                "base_dir": str(tmp_path / "output"),
                "save_final_model": False,
            },
            "wandb": {"dir": str(tmp_path / "logs"), "tags": ["launch-test"]},
        }
    )
    events = []
    observed = {}

    def parse_options(value):
        events.append("parse")
        return checkpoint.parse_checkpoint_options(value)

    def prepare_wandb(value, **kwargs):
        observed["before_prepare"] = deepcopy(value)
        result = checkpoint.prepare_wandb_config(value, **kwargs)
        events.append("prepared")
        observed["prepared"] = result
        return result

    def load_dataset(*args, **kwargs):
        events.append("dataset")
        return LocalDataset(
            [{"prompt": "def solve(x):", "entry_point": "solve", "test": ""}]
        )

    def load_tokenizer(*args, **kwargs):
        events.append("tokenizer")
        return SimpleNamespace(pad_token="pad", eos_token="eos")

    def load_model(*args, **kwargs):
        events.append("model")
        return SimpleNamespace()

    def train(**options):
        events.append("train")
        observed["train_options"] = options

    def create_trainer(**kwargs):
        events.append("construct")
        assert "prepared" in observed, "W&B must be prepared before construction"
        assert kwargs["wandb_config"] is observed["prepared"]
        observed["constructor"] = kwargs
        return SimpleNamespace(train=train)

    monkeypatch.setattr(module, "parse_checkpoint_options", parse_options)
    monkeypatch.setattr(module, "prepare_wandb_config", prepare_wandb)
    monkeypatch.setattr(module, "load_dataset", load_dataset)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", load_tokenizer)
    if hasattr(module, "AutoModelForCausalLM"):
        monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(module, "run_id", lambda: "launch-test")

    import wandb

    monkeypatch.setattr(wandb, "run", None)

    if module_name == "preference_train_common":
        from comlrl.trainers import pbrl

        def invoke():
            module.run_preference_training(
                config=config,
                section_name=section,
                args_cls=getattr(pbrl, trainer_or_config),
                trainer_cls=create_trainer,
                algorithm_name=section,
            )

    else:
        monkeypatch.setattr(module, "Config", lambda *args, **kwargs: config)
        monkeypatch.setattr(module, trainer_or_config, create_trainer)
        monkeypatch.setattr(sys, "argv", [module_name])
        invoke = module.main

    return SimpleNamespace(
        config=config,
        invoke=invoke,
        events=events,
        observed=observed,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize(
    "mode",
    ["default", "resume", "explicit", "disabled-null", "disabled-false", "legacy"],
)
def test_launch_forwards_checkpoint_options_and_prepares_wandb_before_trainer(
    launch, mode, monkeypatch
):
    import comlrl.utils

    options = {}
    if mode != "default":
        options = {
            "resume_from_checkpoint": str(launch.tmp_path / "resume-source"),
            "checkpoint_dir": str(launch.tmp_path / "checkpoint-output"),
            "checkpoint_interval": 3,
            "checkpoint_total_limit": 2,
        }
        launch.config.data["checkpoint"] = dict(options)
    if mode == "explicit":
        launch.config.update({"wandb": {"id": "explicit-run", "resume": "must"}})
    elif mode.startswith("disabled"):
        launch.config.data["wandb"] = None if mode == "disabled-null" else False
    configured_wandb = deepcopy(launch.config.get_section("wandb"))

    def read_metadata(path):
        assert mode in {"resume", "legacy"}, "unexpected checkpoint metadata read"
        assert "construct" not in launch.events
        assert path == options["resume_from_checkpoint"]
        launch.events.append("metadata")
        return {"wandb": None if mode == "legacy" else dict(SAVED_IDENTITY)}

    monkeypatch.setattr(comlrl.utils, "read_checkpoint_metadata", read_metadata)
    launch.invoke()
    assert launch.events[0] == "parse"
    assert launch.events[-1] == "train"
    assert launch.observed["train_options"] == options
    assert not (launch.tmp_path / "resume-source").exists()
    assert not (launch.tmp_path / "checkpoint-output").exists()
    prepared = launch.observed["constructor"]["wandb_config"]
    if mode.startswith("disabled"):
        assert prepared is None
    else:
        for key in ("name", "tags", "dir", "config_sections"):
            assert prepared[key] == launch.observed["before_prepare"][key]
        if mode == "resume":
            for key, value in SAVED_IDENTITY.items():
                assert prepared[key] == value
            assert prepared["resume"] == "allow"
        else:
            assert prepared["project"] == configured_wandb["project"]
            assert prepared["entity"] == configured_wandb["entity"]
            if mode == "explicit":
                assert (prepared["id"], prepared["resume"]) == ("explicit-run", "must")
            else:
                assert "id" not in prepared and "resume" not in prepared
    if mode in {"resume", "legacy"}:
        assert launch.events.count("metadata") == 1
        assert launch.events.index("metadata") < launch.events.index("construct")
    else:
        assert "metadata" not in launch.events


def test_invalid_checkpoint_fails_before_output_data_or_model_work(launch):
    launch.config.data["checkpoint"] = {"checkpoint_interval": True}
    with pytest.raises(ValueError, match="checkpoint_interval"):
        launch.invoke()
    assert launch.events == ["parse"]
    assert not (launch.tmp_path / "output").exists()
