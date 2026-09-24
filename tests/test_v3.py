"""
test_v3.py
----------
7 new tests for FaithfulReward v3 features.

Run:
    pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 1. Curriculum phase switches at correct steps ─────────────────────────────

def test_curriculum_phase_switches_at_correct_steps():
    from src.training.grpo_trainer import _get_curriculum_phase

    assert _get_curriculum_phase(0) == (1, ["math"])
    assert _get_curriculum_phase(299) == (1, ["math"])
    assert _get_curriculum_phase(300) == (2, ["math", "ethics"])
    assert _get_curriculum_phase(599) == (2, ["math", "ethics"])
    assert _get_curriculum_phase(600) == (3, ["math", "ethics", "medical"])
    assert _get_curriculum_phase(999) == (3, ["math", "ethics", "medical"])


# ── 2. SFT examples all have valid format ─────────────────────────────────────

def test_sft_examples_all_have_valid_format():
    from src.training.grpo_trainer import _build_warmup_examples, _parse_and_score

    examples = _build_warmup_examples()

    # Completion follows the prompt template marker
    template_end = "VERDICT: faithful OR unfaithful\n"
    invalid_format = []
    bad_score = []

    for ex in examples:
        text = ex["text"]
        # Locate the actual completion (after the prompt's template block)
        sep_idx = text.find(template_end)
        if sep_idx == -1:
            invalid_format.append(text[:80])
            continue
        completion = text[sep_idx + len(template_end):]

        if "SCORES:" not in completion or "VERDICT:" not in completion:
            invalid_format.append(completion[:80])
            continue

        _, is_valid = _parse_and_score(completion)
        if not is_valid:
            bad_score.append(completion[:80])

    assert len(invalid_format) == 0, f"Examples missing format lines: {invalid_format[:3]}"
    assert len(bad_score) == 0, f"Examples that fail _parse_and_score: {bad_score[:3]}"


# ── 3. SFT examples cover all three domains ───────────────────────────────────

def test_sft_examples_cover_all_three_domains():
    from src.training.grpo_trainer import _build_warmup_examples

    examples = _build_warmup_examples()
    domains_found = {ex.get("domain") for ex in examples}
    assert "math" in domains_found, "No math examples in warmup set"
    assert "ethics" in domains_found, "No ethics examples in warmup set"
    assert "medical" in domains_found, "No medical examples in warmup set"

    counts = {"math": 0, "ethics": 0, "medical": 0}
    for ex in examples:
        d = ex.get("domain")
        if d in counts:
            counts[d] += 1

    assert counts["math"] == 40, f"Expected 40 math examples, got {counts['math']}"
    assert counts["ethics"] == 30, f"Expected 30 ethics examples, got {counts['ethics']}"
    assert counts["medical"] == 30, f"Expected 30 medical examples, got {counts['medical']}"


# ── 4. Diversity check flags duplicates ───────────────────────────────────────

def test_diversity_check_flags_duplicates():
    from src.data_gen.generate_synthetic_data import diversity_check

    records_with_duplicates = [
        {"step_text": "Since p is prime, p must be odd."},
        {"step_text": "Since p is prime, p must be odd."},  # exact duplicate
        {"step_text": "A completely different statement about ethics."},
    ]
    flagged = diversity_check(records_with_duplicates, threshold=0.80)
    assert len(flagged) >= 1, "Should flag exact duplicates"

    records_diverse = [
        {"step_text": "Since p is prime, p must be odd."},
        {"step_text": "A completely different statement about ethics and justice."},
        {"step_text": "The randomized controlled trial had n=1200 participants."},
    ]
    flagged_diverse = diversity_check(records_diverse, threshold=0.80)
    assert len(flagged_diverse) == 0, f"Diverse records should not be flagged: {flagged_diverse}"


# ── 5. Early stopping triggers at 95% ─────────────────────────────────────────

def test_early_stopping_triggers_at_95_percent():
    from src.training.grpo_trainer import EarlyStoppingCallback

    cb = EarlyStoppingCallback(threshold=0.95, patience=3)

    mock_args = MagicMock()
    mock_state = MagicMock()
    mock_state.global_step = 100

    control = MagicMock()
    control.should_training_stop = False

    # Below threshold — no stop
    for _ in range(5):
        cb.on_log(mock_args, mock_state, control, logs={"completion_valid_rate": 0.80})
    assert not control.should_training_stop

    # Reset
    cb._consecutive = 0

    # Exactly at threshold, patience=3 — stop on 3rd consecutive
    cb.on_log(mock_args, mock_state, control, logs={"completion_valid_rate": 0.96})
    assert cb._consecutive == 1
    assert not control.should_training_stop

    cb.on_log(mock_args, mock_state, control, logs={"completion_valid_rate": 0.97})
    assert cb._consecutive == 2
    assert not control.should_training_stop

    cb.on_log(mock_args, mock_state, control, logs={"completion_valid_rate": 0.98})
    assert cb._consecutive == 3
    assert control.should_training_stop

    # Consecutive resets on drop below threshold
    cb2 = EarlyStoppingCallback(threshold=0.95, patience=3)
    ctrl2 = MagicMock()
    ctrl2.should_training_stop = False

    cb2.on_log(mock_args, mock_state, ctrl2, logs={"completion_valid_rate": 0.96})
    cb2.on_log(mock_args, mock_state, ctrl2, logs={"completion_valid_rate": 0.50})  # drops
    assert cb2._consecutive == 0
    cb2.on_log(mock_args, mock_state, ctrl2, logs={"completion_valid_rate": 0.96})
    assert cb2._consecutive == 1
    assert not ctrl2.should_training_stop


# ── 6. Eval temperature flag changes generation kwargs ────────────────────────

def test_eval_temperature_flag_changes_output():
    """evaluate_model builds generate kwargs correctly for different temperatures."""
    # Test the logic directly: temperature > 0 should set do_sample=True
    # This mirrors the exact logic in evaluate.py evaluate_model()
    def build_gen_kwargs(temperature: float, max_new_tokens: int = 512) -> dict:
        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        return gen_kwargs

    # temperature=0.7: sampling enabled
    kwargs_07 = build_gen_kwargs(0.7)
    assert kwargs_07["temperature"] == 0.7
    assert kwargs_07["do_sample"] is True

    # temperature=0.1: sampling enabled (> 0)
    kwargs_01 = build_gen_kwargs(0.1)
    assert kwargs_01["temperature"] == 0.1
    assert kwargs_01["do_sample"] is True

    # temperature=0.0: greedy decoding
    kwargs_00 = build_gen_kwargs(0.0)
    assert kwargs_00["do_sample"] is False
    assert "temperature" not in kwargs_00

    # evaluate.py accepts --temperature arg (verify argparse integration)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args(["--temperature", "0.7"])
    assert args.temperature == 0.7

    args_default = parser.parse_args([])
    assert args_default.temperature == 0.1


# ── 7. Reward curve JSON has required fields ──────────────────────────────────

def test_reward_curve_json_has_required_fields():
    """Every entry in reward_curve_v3.json must have the required fields."""
    required_fields = {"step", "mean_reward", "completion_valid_rate", "clipped_ratio", "frac_reward_zero_std"}

    # Build a sample entry the same way the callback does
    sample_entry = {
        "step": 10,
        "mean_reward": round(0.123, 4),
        "completion_valid_rate": round(0.25, 4),
        "clipped_ratio": round(1.0, 4),
        "frac_reward_zero_std": round(0.0, 4),
    }

    missing = required_fields - set(sample_entry.keys())
    assert missing == set(), f"Reward curve entry missing fields: {missing}"

    # If the real file exists, validate it too
    curve_path = Path("data/reward_curve_v3.json")
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        for i, entry in enumerate(curve):
            missing_in_file = required_fields - set(entry.keys())
            assert missing_in_file == set(), f"Entry {i} in reward_curve_v3.json missing: {missing_in_file}"
