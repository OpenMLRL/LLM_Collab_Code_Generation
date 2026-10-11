import train_grpo
from comlrl.utils.reward_utils import call_reward_function


def test_single_agent_reward_receives_batch_items(monkeypatch):
    seen = {}

    def fake_execution_reward(aux, main, tests, entry_points, prompts):
        seen.update(main=main, tests=tests, entry_points=entry_points)
        return [1.0] * len(main)

    monkeypatch.setattr(train_grpo, "execution_reward_aux", fake_execution_reward)
    item = {"prompt": "def f():", "test": "assert f() == 1", "entry_point": "f"}

    rewards = call_reward_function(
        train_grpo.execution_reward_single_agent,
        ["def f():"],
        [["def f():\n    return 1"]],
        num_agents=1,
        batch_items=[item],
    )

    assert rewards == [1.0]
    assert seen == {
        "main": ["def f():\n    return 1"],
        "tests": ["assert f() == 1"],
        "entry_points": ["f"],
    }
