from pathlib import Path
from types import SimpleNamespace

import pytest

import preference_train_common as setup
from comlrl.trainers import preference
from comlrl.trainers.preference.collaboration import CentralizedCollaboration

ALGORITHMS = [
    ("madpo", preference.MADPOConfig),
    ("marlhf", preference.MARLHFConfig),
    ("madpo_iter", preference.MADPOIterConfig),
    ("marlhf_iter", preference.MARLHFIterConfig),
]


@pytest.mark.parametrize("mode", ["centralized", "decentralized"])
def test_magrpo_entrypoint_selects_joint_actor(mode, monkeypatch, tmp_path):
    import sys
    import train_magrpo as entrypoint

    observed = {}

    def capture(selected_mode, **kwargs):
        observed.update(kwargs)
        observed["selected_mode"] = selected_mode
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(
        entrypoint, "MAGRPOTrainer", lambda **kw: capture("decentralized", **kw)
    )
    monkeypatch.setattr(
        entrypoint,
        "CentralizedMAGRPOTrainer",
        lambda **kw: capture("centralized", **kw),
    )
    item = {"prompt": "def solve(x):", "entry_point": "solve", "test": ""}
    monkeypatch.setattr(entrypoint, "load_dataset", lambda *a, **kw: [item])
    monkeypatch.setattr(
        entrypoint.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(pad_token="pad", eos_token="eos"),
    )
    config = Path(__file__).resolve().parents[1] / "configs/magrpo_che_config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config),
            "--override",
            f"magrpo.collaboration_mode={mode}",
            "magrpo.num_turns=1",
            f"output.base_dir={tmp_path}",
            "output.save_final_model=false",
        ],
    )
    entrypoint.main()
    assert observed["selected_mode"] == mode
    assert observed["num_agents"] == 2
    assert ("centralized_adapter" in observed) == (mode == "centralized")
    assert getattr(observed["args"], "collaboration_mode", "decentralized") == mode


@pytest.mark.parametrize("section,config_cls", ALGORITHMS)
@pytest.mark.parametrize("dataset", ["che"])
@pytest.mark.parametrize("mode", ["decentralized", "centralized"])
def test_preference_setup_passes_collaboration_adapter(
    section, config_cls, dataset, mode, monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    config = setup.Config(str(root / "configs" / f"{section}_{dataset}_config.yaml"))
    assert config.get(f"{section}.collaboration_mode") == "decentralized"
    config.update(
        {
            section: {"collaboration_mode": mode, "agent_devices": "cpu"},
            "output": {"base_dir": str(tmp_path), "save_final_model": False},
        }
    )
    item = {
        "prompt": "A task to solve",
        "abstract_text": "A scientific abstract",
        "entry_point": "solve",
    }
    monkeypatch.setattr(setup, "load_dataset", lambda *a, **kw: [item])
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    monkeypatch.setattr(
        setup.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer
    )
    observed = {}

    def create_trainer(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    setup.run_preference_training(
        config=config,
        section_name=section,
        args_cls=config_cls,
        trainer_cls=create_trainer,
        algorithm_name=section,
    )
    assert observed["num_agents"] == 2
    assert observed["args"].collaboration_mode == mode
    if mode == "centralized":
        runtime = CentralizedCollaboration(
            observed["centralized_comparator_adapter"],
            observed["formatters"],
            lambda left, right: [float(bool(left[0]) and bool(right[0]))],
            2,
        )
        prompt = runtime.build_prompt(item)
        assert "<auxiliary>" in prompt
        assert runtime(
            [
                "<auxiliary>def aux(): return 1</auxiliary><main>def solve(): return aux()</main>"
            ],
            batch_items=[item],
        ) == [1.0]
        if section.endswith("_iter"):
            assert observed["args"].comparator_generation_mode == "centralized"
    elif not section.endswith("_iter"):
        assert "centralized_comparator_adapter" not in observed


@pytest.fixture
def local_models(monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )
    import wandb

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    loaded = []

    def load_model(source, **kwargs):
        loaded.append(source)
        return GPT2LMHeadModel(GPT2Config(vocab_size=3, n_embd=8, n_layer=1, n_head=1))

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(wandb, "init", lambda **kw: None)
    monkeypatch.setattr(wandb, "log", lambda *a, **kw: None)
    return loaded


@pytest.mark.parametrize("section,config_cls", ALGORITHMS)
def test_centralized_preference_loads_one_explicit_actor(
    section, config_cls, local_models, monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[1]
    config = setup.Config(str(root / "configs" / f"{section}_che_config.yaml"))
    options = {"collaboration_mode": "centralized", "agent_devices": "cpu"}
    if section.startswith("marlhf"):
        options["reward_model_device"] = "cpu"
    config.update(
        {
            section: options,
            "agents": ["local-actor"],
            "output": {"base_dir": str(tmp_path), "save_final_model": False},
            "wandb": {"dir": str(tmp_path)},
        }
    )
    item = {"prompt": "def solve(x):", "entry_point": "solve", "test": ""}
    monkeypatch.setattr(setup, "load_dataset", lambda *a, **kw: [item])
    trainer_cls = getattr(preference, config_cls.__name__.replace("Config", "Trainer"))
    observed = []

    def create_trainer(**kwargs):
        trainer = trainer_cls(**kwargs)
        observed.append(trainer)
        trainer.train = lambda: None
        return trainer

    setup.run_preference_training(
        config=config,
        section_name=section,
        args_cls=config_cls,
        trainer_cls=create_trainer,
        algorithm_name=section,
    )
    trainer = observed[0]
    assert local_models == ["local-actor"]
    assert (trainer.num_agents, trainer.args.num_agents) == (1, 2)
    assert len(trainer.agents) == len(trainer.optimizers) == 1
    assert trainer.centralized_collaboration.num_roles == 2


def test_centralized_magrpo_explicit_actor_checkpoint(
    local_models, monkeypatch, tmp_path
):
    import sys
    import train_magrpo as entrypoint

    trainer_cls = entrypoint.CentralizedMAGRPOTrainer
    observed = []

    def create_trainer(**kwargs):
        trainer = trainer_cls(**kwargs)
        observed.append(trainer)
        trainer.train = lambda: None
        return trainer

    monkeypatch.setattr(entrypoint, "CentralizedMAGRPOTrainer", create_trainer)
    monkeypatch.setattr(
        entrypoint,
        "load_dataset",
        lambda *a, **kw: [
            {"prompt": "def solve(x):", "entry_point": "solve", "test": ""}
        ],
    )
    config = Path(__file__).resolve().parents[1] / "configs/magrpo_che_config.yaml"
    checkpoint = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config),
            "--override",
            "magrpo.collaboration_mode=centralized",
            "magrpo.num_turns=1",
            "magrpo.agent_devices=cpu",
            'agents=["local-actor"]',
            f"output.base_dir={tmp_path}",
            "output.save_final_model=true",
            f"output.save_path={checkpoint}",
            f"wandb.dir={tmp_path}",
        ],
    )
    entrypoint.main()
    assert local_models == ["local-actor"]
    assert (observed[0].num_agents, observed[0].args.num_agents) == (1, 2)
    assert (checkpoint / "agent_0" / "config.json").exists()
    assert not (checkpoint / "agent_1").exists()


def test_joint_text_preserves_task_reward_and_eval_metrics():
    from centralized_comparator import CoopHECentralizedComparatorAdapter
    from train_magrpo import (
        get_formatters,
        get_reward_function,
        get_logger_and_aggregator,
    )

    item = {
        "prompt": 'def solve(x):\n    """Add one."""',
        "entry_point": "solve",
        "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    }
    left = "def aux(x):\n    return x + 1"
    right = "def solve(x):\n    result = aux(x)\n    return result"
    reward = get_reward_function("coophumaneval", 2)
    runtime = CentralizedCollaboration(
        CoopHECentralizedComparatorAdapter(),
        get_formatters("coophumaneval", 2),
        reward,
        2,
    )
    joint = f"<auxiliary>{left}</auxiliary><main>{right}</main>"
    assert runtime([joint], batch_items=[item]) == reward(
        [left], [right], batch_items=[item]
    )
    parsed = runtime.split_eval([[[joint]]], [item])
    logger, _ = get_logger_and_aggregator("coophumaneval")
    context = {"batch_items": [item], "prompts": [item["prompt"]]}
    assert logger(agent_completions_turns=parsed, **context) == logger(
        agent_completions_turns=[[[left]], [[right]]], **context
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ('agents=["one", "two"]', "exactly one actor"),
        ("magrpo.num_agents=3", "exactly 2 task roles"),
    ],
)
def test_centralized_magrpo_rejects_wrong_actor_or_role_count(
    override, match, monkeypatch, tmp_path
):
    import sys
    import train_magrpo as entrypoint

    config = Path(__file__).resolve().parents[1] / "configs/magrpo_che_config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config),
            "--override",
            "magrpo.collaboration_mode=centralized",
            f"output.base_dir={tmp_path}",
            override,
        ],
    )
    with pytest.raises(ValueError, match=match):
        entrypoint.main()
