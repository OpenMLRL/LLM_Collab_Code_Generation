"""Bounded joint CoopHE answers in MAAC and the iterative preference methods."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from centralized_comparator import CoopHECentralizedComparatorAdapter
from centralized_native import CoopHENativeCentralizedAdapter
from centralized_stop import (
    MAIN_STOP,
    AuxiliaryPrefillAdapter,
    StopAfterMainMAACTrainer,
    StopAfterMainMADPOIterTrainer,
)
from comlrl.trainers import pbrl
from comlrl.trainers.reinforce import MAGRPOTrainer
from test_centralized_collaboration import _run_preference_setup

ROW = {"prompt": 'def solve(x):\n    """Add one."""', "entry_point": "solve"}
TOKENIZER = SimpleNamespace(
    pad_token_id=0, eos_token_id=0, decode=lambda ids, skip_special_tokens=True: ""
)


def _record_generation(monkeypatch):
    calls = []

    def generate(self, agent, batch_items, **kwargs):
        calls.append(kwargs)
        return {
            "completions": [["<auxiliary>a</auxiliary><main>m</main>"]],
            "completion_input_ids": [[torch.tensor([5, 6])]],
            "completion_attention_mask": [[torch.ones(2)]],
            "response_lens": [2],
        }

    monkeypatch.setattr(MAGRPOTrainer, "_generate_completions", generate)
    return calls


def _preference_trainer(joint_actor):
    trainer = object.__new__(StopAfterMainMADPOIterTrainer)
    trainer.tokenizers = [TOKENIZER, TOKENIZER]
    trainer.num_agents = 2
    trainer.centralized_comparator_adapter = CoopHECentralizedComparatorAdapter()
    trainer.centralized_collaboration = object() if joint_actor else None
    trainer.args = SimpleNamespace(
        max_new_tokens=8,
        comparator_centralized_max_new_tokens=None,
        comparator_centralized_agent_index=0,
    )
    trainer._comparator_rng = lambda agent, **kwargs: nullcontext()
    return trainer


def test_decentralized_rollouts_stay_open_and_the_comparator_is_bounded(
    monkeypatch,
):
    calls = _record_generation(monkeypatch)
    trainer = _preference_trainer(joint_actor=False)
    trainer._generate_completions(None, [ROW], agent_idx=1)
    assert "stop_strings" not in calls[-1]
    trainer._generate_centralized_policy_output_for_agent(
        None, ROW, prompt="joint prompt", num_candidates=1
    )
    assert calls[-1]["stop_strings"] == [MAIN_STOP]
    assert calls[-1]["tokenizer"] is TOKENIZER


def test_a_joint_actor_bounds_every_rollout(monkeypatch):
    calls = _record_generation(monkeypatch)
    _preference_trainer(joint_actor=True)._generate_completions(None, [ROW])
    assert calls[-1]["stop_strings"] == [MAIN_STOP]


def test_bounded_maac_passes_the_stop_to_generate():
    trainer = object.__new__(StopAfterMainMAACTrainer)
    trainer.tokenizers = [TOKENIZER]
    assert trainer._generation_kwargs(0) == {
        "stop_strings": [MAIN_STOP],
        "tokenizer": TOKENIZER,
    }


def _run_maac_entrypoint(monkeypatch, tmp_path, extra, trainer="MAACTrainer"):
    import sys

    import centralized_stop
    import train_maac as entrypoint

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda **options: None)

    def unexpected(**kwargs):
        raise AssertionError("the launcher picked the wrong trainer class")

    plain = trainer == "MAACTrainer"
    monkeypatch.setattr(entrypoint, "MAACTrainer", capture if plain else unexpected)
    monkeypatch.setattr(
        centralized_stop, "StopAfterMainMAACTrainer", unexpected if plain else capture
    )
    monkeypatch.setattr(entrypoint, "load_dataset", lambda *a, **kw: [ROW])
    monkeypatch.setattr(
        entrypoint.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(pad_token="pad", eos_token="eos"),
    )
    config = Path(__file__).resolve().parents[1] / "configs/maac_che_config.yaml"
    argv = [
        "train",
        "--config",
        str(config),
        "--override",
        "maac.num_turns=1",
        f"output.base_dir={tmp_path}",
        "output.save_final_model=false",
        "wandb.enabled=false",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    entrypoint.main()
    return observed


@pytest.mark.parametrize("prompt_mode", ["concatenated", "native"])
@pytest.mark.parametrize("answer", ["open", "bounded"])
def test_maac_entrypoint_builds_one_joint_actor(
    prompt_mode, answer, monkeypatch, tmp_path
):
    observed = _run_maac_entrypoint(
        monkeypatch,
        tmp_path,
        [
            "maac.collaboration_mode=centralized",
            f"maac.centralized_prompt={prompt_mode}",
            f"maac.centralized_answer={answer}",
        ],
        trainer="MAACTrainer" if answer == "open" else "StopAfterMainMAACTrainer",
    )
    assert observed["args"].collaboration_mode == "centralized"
    adapter = observed["centralized_adapter"]
    if answer == "bounded":
        assert type(adapter) is AuxiliaryPrefillAdapter
        adapter = adapter.adapter
    native = prompt_mode == "native"
    assert type(adapter) is (
        CoopHENativeCentralizedAdapter if native else CoopHECentralizedComparatorAdapter
    )


def test_maac_entrypoint_leaves_decentralized_runs_unchanged(monkeypatch, tmp_path):
    observed = _run_maac_entrypoint(monkeypatch, tmp_path, [])
    assert observed["args"].collaboration_mode == "decentralized"
    assert "centralized_adapter" not in observed


@pytest.mark.parametrize(
    "extra,message",
    [
        (["maac.centralized_answer=stop"], "centralized_answer"),
        (["maac.centralized_answer=bounded"], "requires centralized"),
    ],
)
def test_maac_entrypoint_rejects_bad_answer_modes(
    extra, message, monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match=message):
        _run_maac_entrypoint(monkeypatch, tmp_path, extra)


def test_bounded_preference_runs_use_the_stop_trainer(monkeypatch, tmp_path):
    settings = {
        "comparator_generation_mode": "centralized",
        "centralized_prompt": "native",
        "centralized_answer": "bounded",
    }
    observed, _ = _run_preference_setup(
        "madpo_iter",
        pbrl.MADPOIterConfig,
        "che",
        settings,
        monkeypatch,
        tmp_path,
        item=ROW,
        bounded=True,
    )
    adapter = observed["centralized_comparator_adapter"]
    assert type(adapter) is AuxiliaryPrefillAdapter
    assert type(adapter.adapter) is CoopHENativeCentralizedAdapter


@pytest.mark.parametrize(
    "settings,bounded",
    [
        ({"centralized_answer": "bounded"}, True),
        (
            {
                "comparator_generation_mode": "centralized",
                "centralized_answer": "bounded",
            },
            False,
        ),
    ],
)
def test_bounded_preference_runs_need_a_joint_answer_and_a_stop_trainer(
    settings, bounded, monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="centralized_answer=bounded"):
        _run_preference_setup(
            "madpo_iter",
            pbrl.MADPOIterConfig,
            "che",
            settings,
            monkeypatch,
            tmp_path,
            bounded=bounded,
        )


def test_bounded_marlhf_needs_the_magrpo_online_phase(monkeypatch, tmp_path):
    settings = {
        "comparator_generation_mode": "centralized",
        "centralized_answer": "bounded",
        "rl_algorithm": "maac",
    }
    with pytest.raises(ValueError, match="rl_algorithm=magrpo"):
        _run_preference_setup(
            "marlhf_iter",
            pbrl.MARLHFIterConfig,
            "che",
            settings,
            monkeypatch,
            tmp_path,
            bounded=True,
        )
