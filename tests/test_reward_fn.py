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
