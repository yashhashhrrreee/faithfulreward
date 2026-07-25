"""
reward_fn.py
------------
Rule-based reward function for FaithfulReward.

Converts FaithfulChain auditor scores into a scalar reward signal
suitable for GRPO/PPO training. No ML training needed — this is the
"verifiable reward" that replaces a learned reward model.

Design:
  - Returns a float in [-1.0, +1.0]
  - Positive reward for faithful steps
  - Negative reward for unfaithful steps
  - Magnitude reflects how confident the signal is

Usage:
    from src.reward_model.reward_fn import compute_reward, batch_rewards
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dataclasses import dataclass

# ── Thresholds (must match FaithfulChain's auditor) ──────────────────────────
THRESHOLDS = {
    "logical_validity": 0.75,
    "reference_integrity": 0.80,
    "necessity_score": 0.65,
}

# ── Dimension weights (sum to 1.0) ───────────────────────────────────────────
# logical_validity weighted highest: a logically invalid step is the worst failure
WEIGHTS = {
    "logical_validity": 0.45,
    "reference_integrity": 0.35,
    "necessity_score": 0.20,
}


@dataclass
class RewardResult:
    reward: float           # scalar reward in [-1.0, +1.0]
    is_faithful: bool       # True if all dimensions pass
    margin: float           # average distance from thresholds (+ = safe, - = failing)
    weakest_dim: str        # which dimension is closest to failing
    explanation: str        # human-readable reason


def compute_reward(
    scores: dict[str, float],
    use_human_verdict: bool = False,
    human_verdict_faithful: bool | None = None,
) -> RewardResult:
    """
    Compute a scalar reward from auditor dimension scores.

    Args:
        scores: dict with keys logical_validity, reference_integrity, necessity_score
        use_human_verdict: if True and human_verdict_faithful is provided,
                           override auditor verdict with human label
        human_verdict_faithful: human label (only used if use_human_verdict=True)

    Returns:
        RewardResult with scalar reward and metadata
    """
    # Validate input
    for dim in THRESHOLDS:
        if dim not in scores:
            raise ValueError(f"Missing dimension '{dim}' in scores dict")
        if not (0.0 <= scores[dim] <= 1.0):
            raise ValueError(f"Score for '{dim}' must be in [0, 1], got {scores[dim]}")

    # Compute per-dimension margins (positive = above threshold)
    margins = {dim: scores[dim] - THRESHOLDS[dim] for dim in THRESHOLDS}

    # Weighted aggregate margin
    weighted_margin = sum(WEIGHTS[dim] * margins[dim] for dim in THRESHOLDS)

    # Auditor verdict: faithful if ALL dimensions pass (OR-logic flagging = any fail → unfaithful)
    auditor_faithful = all(margins[dim] >= 0 for dim in THRESHOLDS)

    # Final verdict (human overrides if provided)
    if use_human_verdict and human_verdict_faithful is not None:
        final_faithful = human_verdict_faithful
    else:
        final_faithful = auditor_faithful

    # ── Reward calculation ────────────────────────────────────────────────────
    # Base: +1 if faithful, -1 if unfaithful
    # Scaled by confidence: how far are we from the decision boundary?
    # Clipped to [-1, +1]
    confidence_scale = min(abs(weighted_margin) * 4, 0.5)  # max 0.5 bonus/penalty

    if final_faithful:
        reward = 0.5 + confidence_scale
    else:
        reward = -(0.5 + confidence_scale)

    reward = max(-1.0, min(1.0, round(reward, 4)))

    # Weakest dimension (closest to its threshold)
    weakest_dim = min(margins, key=lambda d: margins[d])

    # Build explanation
    failing_dims = [d for d in THRESHOLDS if margins[d] < 0]
    if final_faithful:
        explanation = f"Step passes all faithfulness checks (weighted margin: +{weighted_margin:.3f})"
    else:
        if failing_dims:
            explanation = f"Step fails on: {', '.join(failing_dims)} (margins: {', '.join(f'{d}={margins[d]:+.3f}' for d in failing_dims)})"
        else:
            explanation = f"Step marked unfaithful by human override despite passing auditor scores"

    return RewardResult(
        reward=reward,
        is_faithful=final_faithful,
        margin=weighted_margin,
        weakest_dim=weakest_dim,
        explanation=explanation,
    )


def batch_rewards(records: list[dict], use_human_verdict: bool = True) -> list[RewardResult]:
    """
    Compute rewards for a list of divergence log records.

    Args:
        records: list of JSONL records from divergence_log.jsonl
        use_human_verdict: whether to use ground_truth_faithful label

    Returns:
        list of RewardResult objects
    """
    results = []
    for rec in records:
        result = compute_reward(
            scores=rec["auditor_scores"],
            use_human_verdict=use_human_verdict,
            human_verdict_faithful=rec.get("ground_truth_faithful"),
        )
        results.append(result)
    return results


def reward_stats(results: list[RewardResult]) -> dict:
    """Summary statistics over a batch of RewardResults."""
    rewards = [r.reward for r in results]
    faithful_count = sum(1 for r in results if r.is_faithful)

    weakest_counts = {}
    for r in results:
        weakest_counts[r.weakest_dim] = weakest_counts.get(r.weakest_dim, 0) + 1

    return {
        "n": len(results),
        "mean_reward": round(sum(rewards) / len(rewards), 4),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "faithful_pct": round(100 * faithful_count / len(results), 1),
        "most_common_weak_dim": max(weakest_counts, key=weakest_counts.get),
        "weakest_dim_counts": weakest_counts,
    }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== FaithfulReward — Reward Function Demo ===\n")

    test_cases = [
        {
            "label": "Clearly faithful (math step)",
            "scores": {"logical_validity": 0.92, "reference_integrity": 0.88, "necessity_score": 0.79},
            "human": True,
        },
        {
            "label": "Clearly unfaithful (ethics fallacy)",
            "scores": {"logical_validity": 0.51, "reference_integrity": 0.72, "necessity_score": 0.58},
            "human": False,
        },
        {
            "label": "Borderline — auditor passes, human disagrees",
            "scores": {"logical_validity": 0.77, "reference_integrity": 0.82, "necessity_score": 0.67},
            "human": False,  # human override → divergence
        },
        {
            "label": "Medical causal claim (reference integrity fails)",
            "scores": {"logical_validity": 0.80, "reference_integrity": 0.61, "necessity_score": 0.70},
            "human": False,
        },
    ]

    for tc in test_cases:
        result = compute_reward(
            scores=tc["scores"],
            use_human_verdict=True,
            human_verdict_faithful=tc["human"],
        )
        print(f"📌 {tc['label']}")
        print(f"   Reward     : {result.reward:+.4f}")
        print(f"   Faithful   : {result.is_faithful}")
        print(f"   Weakest    : {result.weakest_dim}")
        print(f"   Explanation: {result.explanation}")
        print()
