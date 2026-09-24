"""
test_reward_fn.py
-----------------
25 pytest tests for the FaithfulReward reward function.

Run:
    pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reward_model.reward_fn import compute_reward, batch_rewards, reward_stats, RewardResult

# ── Fixtures ──────────────────────────────────────────────────────────────────

PASSING_SCORES = {"logical_validity": 0.90, "reference_integrity": 0.88, "necessity_score": 0.80}
FAILING_SCORES = {"logical_validity": 0.50, "reference_integrity": 0.60, "necessity_score": 0.45}
BORDERLINE_PASS = {"logical_validity": 0.76, "reference_integrity": 0.81, "necessity_score": 0.66}
BORDERLINE_FAIL = {"logical_validity": 0.74, "reference_integrity": 0.79, "necessity_score": 0.64}
ONE_DIM_FAIL = {"logical_validity": 0.90, "reference_integrity": 0.70, "necessity_score": 0.80}  # ref_integrity fails


# ── Return type tests ─────────────────────────────────────────────────────────

def test_returns_reward_result():
    result = compute_reward(PASSING_SCORES)
    assert isinstance(result, RewardResult)

def test_reward_is_float():
    result = compute_reward(PASSING_SCORES)
    assert isinstance(result.reward, float)

def test_reward_bounded_upper():
    result = compute_reward(PASSING_SCORES)
    assert result.reward <= 1.0

def test_reward_bounded_lower():
    result = compute_reward(FAILING_SCORES)
    assert result.reward >= -1.0

def test_reward_result_has_explanation():
    result = compute_reward(PASSING_SCORES)
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0


# ── Faithful / unfaithful direction ───────────────────────────────────────────

def test_passing_scores_give_positive_reward():
    result = compute_reward(PASSING_SCORES)
    assert result.reward > 0

def test_failing_scores_give_negative_reward():
    result = compute_reward(FAILING_SCORES)
    assert result.reward < 0

def test_passing_scores_mark_as_faithful():
    result = compute_reward(PASSING_SCORES)
    assert result.is_faithful is True

def test_failing_scores_mark_as_unfaithful():
    result = compute_reward(FAILING_SCORES)
    assert result.is_faithful is False

def test_borderline_pass_is_faithful():
    result = compute_reward(BORDERLINE_PASS)
    assert result.is_faithful is True

def test_borderline_fail_is_unfaithful():
    result = compute_reward(BORDERLINE_FAIL)
    assert result.is_faithful is False


# ── OR-logic: single failing dimension ───────────────────────────────────────

def test_single_failing_dim_makes_unfaithful():
    """OR-logic: any dimension below threshold → unfaithful."""
    result = compute_reward(ONE_DIM_FAIL)
    assert result.is_faithful is False

def test_single_failing_dim_gives_negative_reward():
    result = compute_reward(ONE_DIM_FAIL)
    assert result.reward < 0

def test_weakest_dim_identified_correctly():
    result = compute_reward(ONE_DIM_FAIL)
    assert result.weakest_dim == "reference_integrity"


# ── Human verdict override ────────────────────────────────────────────────────

def test_human_override_faithful_changes_verdict():
    """Auditor would flag, but human says faithful."""
    result = compute_reward(FAILING_SCORES, use_human_verdict=True, human_verdict_faithful=True)
    assert result.is_faithful is True
    assert result.reward > 0

def test_human_override_unfaithful_changes_verdict():
    """Auditor would pass, but human says unfaithful."""
    result = compute_reward(PASSING_SCORES, use_human_verdict=True, human_verdict_faithful=False)
    assert result.is_faithful is False
    assert result.reward < 0

def test_human_override_ignored_when_flag_false():
    """use_human_verdict=False means auditor verdict wins."""
    result = compute_reward(FAILING_SCORES, use_human_verdict=False, human_verdict_faithful=True)
    assert result.is_faithful is False  # auditor wins


# ── Confidence scaling ────────────────────────────────────────────────────────

def test_confident_faithful_reward_higher_than_borderline():
    confident = compute_reward(PASSING_SCORES)
    borderline = compute_reward(BORDERLINE_PASS)
    assert confident.reward > borderline.reward

def test_confident_unfaithful_reward_lower_than_borderline():
    confident = compute_reward(FAILING_SCORES)
    borderline = compute_reward(BORDERLINE_FAIL)
    assert confident.reward < borderline.reward


# ── Input validation ──────────────────────────────────────────────────────────

def test_missing_dimension_raises_value_error():
    incomplete = {"logical_validity": 0.90, "reference_integrity": 0.85}
    with pytest.raises(ValueError, match="Missing dimension"):
        compute_reward(incomplete)

def test_score_out_of_range_raises_value_error():
    bad_scores = {"logical_validity": 1.5, "reference_integrity": 0.85, "necessity_score": 0.70}
    with pytest.raises(ValueError, match="must be in"):
        compute_reward(bad_scores)

def test_negative_score_raises_value_error():
    bad_scores = {"logical_validity": -0.1, "reference_integrity": 0.85, "necessity_score": 0.70}
    with pytest.raises(ValueError, match="must be in"):
        compute_reward(bad_scores)


# ── Batch functions ───────────────────────────────────────────────────────────

def test_batch_rewards_returns_list():
    records = [
        {"auditor_scores": PASSING_SCORES, "ground_truth_faithful": True},
        {"auditor_scores": FAILING_SCORES, "ground_truth_faithful": False},
    ]
    results = batch_rewards(records)
    assert isinstance(results, list)
    assert len(results) == 2

def test_batch_rewards_correct_types():
    records = [{"auditor_scores": PASSING_SCORES, "ground_truth_faithful": True}]
    results = batch_rewards(records)
    assert isinstance(results[0], RewardResult)

def test_reward_stats_structure():
    records = [
        {"auditor_scores": PASSING_SCORES, "ground_truth_faithful": True},
        {"auditor_scores": FAILING_SCORES, "ground_truth_faithful": False},
        {"auditor_scores": ONE_DIM_FAIL, "ground_truth_faithful": False},
    ]
    results = batch_rewards(records)
    stats = reward_stats(results)
    assert "mean_reward" in stats
    assert "faithful_pct" in stats
    assert "most_common_weak_dim" in stats
    assert stats["n"] == 3


# ── Format reward bonus tests ─────────────────────────────────────────────────

from src.reward_model.reward_fn import format_reward_bonus

VALID_COMPLETION = (
    "STEP: Since p is prime, p must be odd.\n"
    "SCORES: logical_validity=0.90, reference_integrity=0.88, necessity_score=0.80\n"
    "VERDICT: faithful"
)
MISSING_VERDICT_COMPLETION = (
    "STEP: Since p is prime, p must be odd.\n"
    "SCORES: logical_validity=0.90, reference_integrity=0.88, necessity_score=0.80"
)
MISSING_SCORES_COMPLETION = (
    "STEP: Since p is prime, p must be odd.\n"
    "VERDICT: faithful"
)
MALFORMED_COMPLETION = "I think this step is probably fine based on intuition."


def test_format_reward_bonus_applies_for_valid_completion():
    result = format_reward_bonus(VALID_COMPLETION, 0.5)
    assert abs(result - 0.6) < 1e-4


def test_format_reward_bonus_clips_to_one():
    result = format_reward_bonus(VALID_COMPLETION, 1.0)
    assert result == 1.0


def test_format_reward_bonus_negative_base_still_gets_bonus():
    result = format_reward_bonus(VALID_COMPLETION, -0.5)
    assert abs(result - (-0.4)) < 1e-4


def test_format_reward_bonus_no_bonus_missing_verdict():
    result = format_reward_bonus(MISSING_VERDICT_COMPLETION, 0.5)
    assert result == 0.5


def test_format_reward_bonus_no_bonus_missing_scores():
    result = format_reward_bonus(MISSING_SCORES_COMPLETION, 0.5)
    assert result == 0.5


def test_format_reward_bonus_no_bonus_malformed():
    result = format_reward_bonus(MALFORMED_COMPLETION, -0.3)
    assert result == -0.3


# ── completion_valid_rate tests ───────────────────────────────────────────────

def test_completion_valid_rate_tracks_valid_completions():
    from src.training.grpo_trainer import faithfulness_reward_fn, _valid_completions_window
    _valid_completions_window.clear()

    completions = [VALID_COMPLETION, MISSING_VERDICT_COMPLETION, VALID_COMPLETION, MALFORMED_COMPLETION]
    faithfulness_reward_fn(completions)

    valid_count = sum(_valid_completions_window)
    assert valid_count == 2  # 2 of 4 completions are fully valid


def test_completion_valid_rate_zero_for_all_malformed():
    from src.training.grpo_trainer import faithfulness_reward_fn, _valid_completions_window
    _valid_completions_window.clear()

    completions = [MALFORMED_COMPLETION, "no format", "garbage"]
    faithfulness_reward_fn(completions)

    assert sum(_valid_completions_window) == 0


def test_completion_valid_rate_one_for_all_valid():
    from src.training.grpo_trainer import faithfulness_reward_fn, _valid_completions_window
    _valid_completions_window.clear()

    completions = [VALID_COMPLETION, VALID_COMPLETION, VALID_COMPLETION]
    faithfulness_reward_fn(completions)

    rate = sum(_valid_completions_window) / len(_valid_completions_window)
    assert abs(rate - 1.0) < 1e-9


# ── format_warmup tests ───────────────────────────────────────────────────────

def test_build_warmup_examples_returns_100():
    from src.training.grpo_trainer import _build_warmup_examples
    examples = _build_warmup_examples()
    assert len(examples) == 100


def test_warmup_examples_all_have_scores_line():
    from src.training.grpo_trainer import _build_warmup_examples
    examples = _build_warmup_examples()
    for ex in examples:
        assert "SCORES:" in ex["text"], f"Missing SCORES: in example: {ex['text'][:80]}"


def test_warmup_examples_all_have_verdict_line():
    from src.training.grpo_trainer import _build_warmup_examples
    examples = _build_warmup_examples()
    for ex in examples:
        assert "VERDICT:" in ex["text"], f"Missing VERDICT: in example: {ex['text'][:80]}"


def test_format_warmup_callable():
    from src.training.grpo_trainer import format_warmup
    assert callable(format_warmup)


def test_format_warmup_with_mocked_trainer():
    """format_warmup should call SFTTrainer.train() and return sft_trainer.model."""
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_lora_config = MagicMock()
    mock_trained_model = MagicMock()

    mock_sft_instance = MagicMock()
    mock_sft_instance.model = mock_trained_model

    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_list.return_value = MagicMock()

    mock_trl = MagicMock()
    mock_trl.SFTConfig.return_value = MagicMock()
    mock_trl.SFTTrainer.return_value = mock_sft_instance

    with patch.dict("sys.modules", {"trl": mock_trl, "datasets": MagicMock(Dataset=mock_dataset_cls)}):
        import importlib
        import src.training.grpo_trainer as trainer_module
        importlib.reload(trainer_module)

        result = trainer_module.format_warmup(
            mock_model, mock_tokenizer, mock_lora_config, warmup_steps=1
        )

    mock_sft_instance.train.assert_called_once()
    assert result == mock_trained_model
