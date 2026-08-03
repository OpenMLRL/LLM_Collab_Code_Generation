from centralized_comparator import CoopHECentralizedComparatorAdapter


def test_coophe_prompt_preserves_auxiliary_main_protocol():
    adapter = CoopHECentralizedComparatorAdapter()
    prompt = adapter.build_prompt(
        {
            "entry_point": "solve",
            "prompt": "def solve(value):\n    pass",
        },
        ["auxiliary prompt", "main prompt"],
    )
    assert "Auxiliary agent original prompt" in prompt
    assert "<auxiliary>" in prompt
    assert "def solve(value):" in prompt


def test_coophe_parser_preserves_partial_output_fallback():
    outputs = CoopHECentralizedComparatorAdapter().parse_completion(
        "<main>\ndef solve(value):\n    return value\n</main>",
        {"entry_point": "solve"},
        2,
    )
    assert outputs == ["", "def solve(value):\n    return value"]
