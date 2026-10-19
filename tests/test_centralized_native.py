from pathlib import Path
from types import SimpleNamespace

import pytest

from comlrl.trainers.preference.collaboration import CentralizedCollaboration

ITEM = {
    "prompt": 'def solve(x):\n    """Add one."""',
    "entry_point": "solve",
    "test": "def check(candidate):\n    assert candidate(1) == 2\n",
    "canonical_solution": "SECRET_GOLD",
}
LEFT = "def aux(x):\n    return x + 1"
RIGHT = "def solve(x):\n    result = aux(x)\n    return result"


def _runtime(adapter):
    from train_magrpo import get_formatters, get_reward_function

    return CentralizedCollaboration(
        adapter,
        get_formatters("coophumaneval", 2),
        get_reward_function("coophumaneval", 2),
        2,
    )


def test_native_prompt_states_the_problem_once():
    from centralized_comparator import CoopHECentralizedComparatorAdapter
    from centralized_native import CoopHENativeCentralizedAdapter

    prompt = _runtime(CoopHENativeCentralizedAdapter()).build_prompt(ITEM)
    assert prompt.count(ITEM["prompt"]) == 1
    assert "SECRET_GOLD" not in prompt
    assert "original prompt" not in prompt
    assert "<auxiliary>" in prompt and "<main>" in prompt
    assert "def solve(x):" in prompt
    concatenated = _runtime(CoopHECentralizedComparatorAdapter()).build_prompt(ITEM)
    assert concatenated.count(ITEM["prompt"]) == 2
    assert len(prompt) < len(concatenated)


def test_native_prompt_keeps_the_role_formatters_error_rows():
    from centralized_native import build_native_centralized_prompt
    from train_magrpo import main_function_formatter

    for item in ({**ITEM, "entry_point": ""}, {**ITEM, "prompt": "def solve():"}):
        assert build_native_centralized_prompt(item) == main_function_formatter(item)


def test_native_split_gives_the_unchanged_reward_and_logs():
    from centralized_native import CoopHENativeCentralizedAdapter
    from train_magrpo import get_logger_and_aggregator, get_reward_function

    reward = get_reward_function("coophumaneval", 2)
    runtime = _runtime(CoopHENativeCentralizedAdapter())
    joint = f"<auxiliary>{LEFT}</auxiliary><main>{RIGHT}</main>"
    assert runtime([joint], batch_items=[ITEM]) == reward(
        [LEFT], [RIGHT], batch_items=[ITEM]
    )
    parsed = runtime.split_eval([[[joint]]], [ITEM])
    logger, _ = get_logger_and_aggregator("coophumaneval")
    context = {"batch_items": [ITEM], "prompts": [ITEM["prompt"]]}
    assert logger(agent_completions_turns=parsed, **context) == logger(
        agent_completions_turns=[[[LEFT]], [[RIGHT]]], **context
    )


def _run_entrypoint(monkeypatch, tmp_path, extra, trainer="CentralizedMAGRPOTrainer"):
    import sys
    import train_magrpo as entrypoint

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(train=lambda: None)

    def unexpected(**kwargs):
        raise AssertionError("the launcher picked the wrong trainer class")

    for name in ("CentralizedMAGRPOTrainer", "StopAfterMainMAGRPOTrainer"):
        monkeypatch.setattr(entrypoint, name, capture if name == trainer else unexpected)
    monkeypatch.setattr(entrypoint, "load_dataset", lambda *a, **kw: [ITEM])
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
            "magrpo.collaboration_mode=centralized",
            "magrpo.num_turns=1",
            f"output.base_dir={tmp_path}",
            "output.save_final_model=false",
            *extra,
        ],
    )
    entrypoint.main()
    return observed


@pytest.mark.parametrize(
    "extra,adapter_name",
    [
        ([], "CoopHECentralizedComparatorAdapter"),
        (
            ["magrpo.centralized_prompt=concatenated"],
            "CoopHECentralizedComparatorAdapter",
        ),
        (["magrpo.centralized_prompt=native"], "CoopHENativeCentralizedAdapter"),
    ],
)
def test_entrypoint_selects_the_centralized_prompt(
    extra, adapter_name, monkeypatch, tmp_path
):
    observed = _run_entrypoint(monkeypatch, tmp_path, extra)
    assert type(observed["centralized_adapter"]).__name__ == adapter_name
    # The default config shares the decentralized loss and skips empty updates.
    assert observed["args"].match_decentralized_loss is True
    assert observed["args"].skip_zero_grad_steps is True


def test_entrypoint_rejects_unknown_prompt_mode(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="centralized_prompt"):
        _run_entrypoint(monkeypatch, tmp_path, ["magrpo.centralized_prompt=joint"])


def _adapter(name):
    from centralized_comparator import CoopHECentralizedComparatorAdapter
    from centralized_native import CoopHENativeCentralizedAdapter

    return {
        "concatenated": CoopHECentralizedComparatorAdapter,
        "native": CoopHENativeCentralizedAdapter,
    }[name]()


@pytest.mark.parametrize("prompt_mode", ["concatenated", "native"])
def test_bounded_answer_starts_inside_auxiliary_with_the_same_reward(prompt_mode):
    from centralized_stop import AuxiliaryPrefillAdapter
    from train_magrpo import get_reward_function

    inner = _adapter(prompt_mode)
    runtime = _runtime(AuxiliaryPrefillAdapter(inner))
    assert runtime.build_prompt(ITEM) == _runtime(inner).build_prompt(ITEM) + (
        "<auxiliary>\n"
    )
    # Generation returns only the text after the prefilled opening tag.
    completion = f"{LEFT}\n</auxiliary>\n<main>\n{RIGHT}\n</main>"
    assert runtime.split(completion, ITEM) == [LEFT, RIGHT]
    assert runtime.split("<auxiliary>\n" + completion, ITEM) == [LEFT, RIGHT]
    reward = get_reward_function("coophumaneval", 2)
    assert runtime([completion], batch_items=[ITEM]) == reward(
        [LEFT], [RIGHT], batch_items=[ITEM]
    )


def test_stop_trainer_stops_at_main_and_keeps_only_sampled_tokens(monkeypatch):
    import torch
    from comlrl.trainers.reinforce import CentralizedMAGRPOTrainer
    from centralized_stop import StopAfterMainMAGRPOTrainer

    pieces = {1: LEFT + "\n</auxiliary>\n<main>\n", 2: RIGHT + "\n", 3: "</main>", 4: " prose"}
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=0,
        decode=lambda ids, skip_special_tokens=True: "".join(
            pieces[int(i)] for i in ids if int(i)
        ),
    )
    stopped = torch.tensor([1, 2, 3, 0, 0])  # reached </main>, then padded
    ended = torch.tensor([1, 2, 0, 0, 0])  # sampled EOS before </main>
    capped = torch.tensor([1, 2, 4, 4, 4])  # ran out of tokens
    seen = {}

    def generate(self, agent, batch_items, **kwargs):
        seen.update(kwargs)
        sequences = [stopped, ended, capped]
        return {
            "completions": [["stopped", "ended", "capped"]],
            "completion_input_ids": [sequences],
            "completion_attention_mask": [[torch.ones(5) for _ in sequences]],
            # CoMLRL counts the first pad as a sampled EOS when pad == EOS.
            "response_lens": [4, 3, 5],
        }

    monkeypatch.setattr(CentralizedMAGRPOTrainer, "_generate_completions", generate)
    trainer = object.__new__(StopAfterMainMAGRPOTrainer)
    trainer.tokenizers = [tokenizer]
    trainer.args = SimpleNamespace(match_decentralized_loss=False)
    result = trainer._generate_completions(None, [ITEM], num_return_sequences=3)
    assert seen["stop_strings"] == ["</main>"] and seen["tokenizer"] is tokenizer
    assert [t.tolist() for t in result["completion_input_ids"][0]] == [
        [1, 2, 3],
        ended.tolist(),
        capped.tolist(),
    ]
    assert result["response_lens"] == [3, 3, 5]
    assert [len(m) for m in result["completion_attention_mask"][0]] == [3, 5, 5]
    assert result["completions"] == [["stopped", "ended", "capped"]]
    # With the decentralized loss, rows keep their padding like decentralized rows.
    trainer.args = SimpleNamespace(match_decentralized_loss=True)
    result = trainer._generate_completions(None, [ITEM], num_return_sequences=3)
    assert [t.tolist() for t in result["completion_input_ids"][0]] == [
        stopped.tolist(),
        ended.tolist(),
        capped.tolist(),
    ]
    assert result["response_lens"] == [4, 3, 5]


@pytest.mark.parametrize("prompt_mode", ["concatenated", "native"])
def test_entrypoint_bounded_answer_uses_the_stop_trainer(
    prompt_mode, monkeypatch, tmp_path
):
    observed = _run_entrypoint(
        monkeypatch,
        tmp_path,
        [f"magrpo.centralized_prompt={prompt_mode}", "magrpo.centralized_answer=bounded"],
        trainer="StopAfterMainMAGRPOTrainer",
    )
    adapter = observed["centralized_adapter"]
    assert type(adapter).__name__ == "AuxiliaryPrefillAdapter"
    assert type(adapter.adapter) is type(_adapter(prompt_mode))


@pytest.mark.parametrize(
    "extra,message",
    [
        (["magrpo.centralized_answer=stop"], "centralized_answer"),
        (
            ["magrpo.collaboration_mode=decentralized", "magrpo.centralized_answer=bounded"],
            "requires centralized",
        ),
    ],
)
def test_entrypoint_rejects_bad_answer_modes(extra, message, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match=message):
        _run_entrypoint(monkeypatch, tmp_path, extra)
