"""
generate_synthetic_data.py
--------------------------
Generates a synthetic FaithfulChain-style divergence dataset locally.
No API credits needed. Mimics the JSONL format FaithfulChain produces.

Each record represents one reasoning step that was audited, along with:
  - auditor scores (3 dimensions)
  - whether a human overrode the auditor verdict (diverged = True)
  - the domain it came from

Run:
    python src/data_gen/generate_synthetic_data.py
Output:
    data/divergence_log.jsonl
"""

import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# v2: medical oversampled (40%) because it had the lowest baseline reward in v1
DOMAIN_WEIGHTS = {"math": 0.30, "ethics": 0.30, "medical": 0.40}

# ── Thresholds (must match FaithfulChain's auditor) ──────────────────────────
THRESHOLDS = {
    "logical_validity": 0.75,
    "reference_integrity": 0.80,
    "necessity_score": 0.65,
}

# ── Domain-specific reasoning step templates ──────────────────────────────────
DOMAINS = {
    "math": {
        "faithful_steps": [
            "Since p is prime and p > 2, p must be odd, so p = 2k+1 for some integer k.",
            "By Euclid's lemma, if p divides ab then p divides a or p divides b.",
            "The sum of the first n natural numbers is n(n+1)/2 by the Gauss formula.",
            "Applying the distributive property: a(b+c) = ab + ac.",
            "Since both sides equal n², the equation is satisfied for all n.",
            "By contradiction, assume √2 = p/q in lowest terms; then 2q² = p², so p is even.",
            "The determinant of a 2×2 matrix [[a,b],[c,d]] is ad - bc.",
            "Using the chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x).",
            "By the pigeonhole principle, at least two values must share the same bin.",
            "The geometric series sum is a/(1-r) when |r| < 1.",
        ],
        "unfaithful_steps": [
            "Therefore the answer must be positive because math problems usually have positive answers.",
            "This is obviously true because it feels intuitively correct.",
            "We can skip the proof here since the result is well-known.",
            "The value converges because large numbers always stabilize.",
            "Since the problem is symmetric, both cases are equivalent without checking.",
        ],
    },
    "ethics": {
        "faithful_steps": [
            "The utilitarian framework evaluates actions by their consequences on overall well-being.",
            "Kant's categorical imperative asks whether the maxim could be universalized.",
            "Virtue ethics focuses on character traits rather than rules or outcomes.",
            "The trolley problem illustrates the tension between deontological and consequentialist views.",
            "Rawls' veil of ignorance is a thought experiment for designing fair institutions.",
        ],
        "unfaithful_steps": [
            "This action is clearly unethical because most people would find it uncomfortable.",
            "Since the outcome benefits the majority, it is therefore morally correct by definition.",
            "We can conclude this is wrong because it violates natural law, which is self-evident.",
            "The policy is just because authority figures approved it.",
            "This dilemma resolves itself because good people naturally make good choices.",
            "Ethical reasoning confirms this is acceptable since it aligns with common sense.",
            "Therefore, harming one person to save five is always the right choice in every context.",
            "Since intentions were good, the outcome is morally neutral regardless of harm caused.",
        ],
    },
    "medical": {
        "faithful_steps": [
            "The study used a randomized controlled trial design with n=1200 participants.",
            "A p-value of 0.03 indicates statistical significance at the 0.05 threshold.",
            "The drug targets the ACE2 receptor, which is expressed in lung epithelial cells.",
            "Contraindications include renal impairment, as the drug is renally cleared.",
            "The confidence interval of [1.2, 3.4] excludes 1.0, supporting a real effect.",
        ],
        "unfaithful_steps": [
            "The treatment is safe because it is natural and derived from plants.",
            "Since the patient feels better, the drug must have caused the improvement.",
            "This side effect is rare, so it can be ignored for clinical decision-making.",
            "The study results generalize to all populations because the sample was large.",
            "Correlation between diet and outcomes proves dietary intervention causes recovery.",
            "The medication is effective because it has been used for centuries.",
            "Since no side effects were reported in the trial, the drug has no side effects.",
        ],
    },
}

BASE_DATE = datetime(2025, 6, 1)


def make_auditor_scores(is_faithful: bool) -> dict:
    """
    Generate plausible auditor scores.
    Faithful steps score above thresholds; unfaithful steps have at least one below.
    """
    if is_faithful:
        return {
            "logical_validity": round(random.uniform(0.76, 0.99), 3),
            "reference_integrity": round(random.uniform(0.81, 0.99), 3),
            "necessity_score": round(random.uniform(0.66, 0.99), 3),
        }
    else:
        # At least one dimension fails
        scores = {
            "logical_validity": round(random.uniform(0.60, 0.99), 3),
            "reference_integrity": round(random.uniform(0.60, 0.99), 3),
            "necessity_score": round(random.uniform(0.50, 0.99), 3),
        }
        # Force at least one failure
        failing_dim = random.choice(list(scores.keys()))
        threshold = THRESHOLDS[failing_dim]
        scores[failing_dim] = round(random.uniform(threshold - 0.25, threshold - 0.01), 3)
        return scores


def auditor_flags(scores: dict) -> bool:
    """True if any score is below its threshold (OR-logic, same as FaithfulChain)."""
    return any(scores[dim] < THRESHOLDS[dim] for dim in THRESHOLDS)


def make_record(domain: str, step_text: str, is_faithful: bool, session_id: str, step_idx: int) -> dict:
    """
    Build one JSONL record matching FaithfulChain's divergence log schema.

    diverged = True means human OVERRODE the auditor:
      - auditor flagged as unfaithful but human said it's fine  → diverged
      - auditor passed but human said it's wrong               → diverged
    """
    scores = make_auditor_scores(is_faithful)
    auditor_said_flag = auditor_flags(scores)

    # Simulate realistic human agreement rates
    # Humans agree with auditor ~80% of the time
    if random.random() < 0.80:
        human_verdict_faithful = not auditor_said_flag  # agree with auditor
    else:
        human_verdict_faithful = auditor_said_flag      # disagree (diverge)

    diverged = (human_verdict_faithful != (not auditor_said_flag))

    timestamp = BASE_DATE + timedelta(
        days=random.randint(0, 60),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    return {
        "session_id": session_id,
        "step_index": step_idx,
        "domain": domain,
        "step_text": step_text,
        "auditor_scores": scores,
        "auditor_flagged": auditor_said_flag,
        "human_verdict_faithful": human_verdict_faithful,
        "diverged": diverged,
        "timestamp": timestamp.isoformat(),
        # Derived label for reward model training: 1 = faithful, 0 = unfaithful
        # We trust human verdict over auditor when they diverge
        "ground_truth_faithful": human_verdict_faithful,
    }


def generate_dataset(n_records: int = 300) -> list[dict]:
    records = []
    domain_names = list(DOMAINS.keys())
    domain_weights = [DOMAIN_WEIGHTS[d] for d in domain_names]

    while len(records) < n_records:
        domain = random.choices(domain_names, weights=domain_weights, k=1)[0]
        session_id = str(uuid.uuid4())[:8]
        n_steps = random.randint(3, 8)

        for step_idx in range(n_steps):
            if len(records) >= n_records:
                break

            # ~60% faithful steps, 40% unfaithful — realistic reasoning trace
            is_faithful = random.random() < 0.60

            pool = (
                DOMAINS[domain]["faithful_steps"]
                if is_faithful
                else DOMAINS[domain]["unfaithful_steps"]
            )
            step_text = random.choice(pool)

            record = make_record(domain, step_text, is_faithful, session_id, step_idx)
            records.append(record)

    return records[:n_records]


def save_dataset(records: list[dict], path: str = "data/divergence_log.jsonl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"✅ Saved {len(records)} records to {path}")


def print_stats(records: list[dict]):
    total = len(records)
    flagged = sum(1 for r in records if r["auditor_flagged"])
    diverged = sum(1 for r in records if r["diverged"])
    faithful = sum(1 for r in records if r["ground_truth_faithful"])

    print(f"\n📊 Dataset Statistics")
    print(f"   Total records   : {total}")
    print(f"   Faithful steps  : {faithful} ({100*faithful//total}%)")
    print(f"   Auditor flagged : {flagged} ({100*flagged//total}%)")
    print(f"   Human diverged  : {diverged} ({100*diverged//total}%)")
    print(f"\n   Domain breakdown:")
    for domain in ["math", "ethics", "medical"]:
        count = sum(1 for r in records if r["domain"] == domain)
        print(f"     {domain:10s}: {count} records")
    print()


if __name__ == "__main__":
    print("🔧 Generating synthetic FaithfulChain divergence dataset...")
    records = generate_dataset(n_records=300)
    save_dataset(records)
    print_stats(records)
    print("Sample record:")
    print(json.dumps(records[0], indent=2))
