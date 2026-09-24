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
            # 15 new faithful math steps
            "The integral of x^n is x^(n+1)/(n+1) + C for n ≠ -1 by the power rule.",
            "A matrix is invertible iff its determinant is non-zero.",
            "By induction: if P(1) holds and P(k)→P(k+1), then P(n) holds for all n≥1.",
            "Since f is continuous on [a,b] and differentiable on (a,b), MVT gives f'(c)=(f(b)-f(a))/(b-a).",
            "The Pythagorean theorem states a²+b²=c² for a right triangle with hypotenuse c.",
            "log(ab) = log(a) + log(b) by the product rule of logarithms.",
            "A function f is bijective iff it has an inverse function f⁻¹.",
            "The binomial coefficient C(n,k) counts k-element subsets of an n-element set.",
            "By De Morgan's law, ¬(A∧B) = ¬A∨¬B.",
            "The rank-nullity theorem states rank(A) + nullity(A) = n for an m×n matrix A.",
            "Since the discriminant b²-4ac < 0, the quadratic has no real roots.",
            "Euler's formula states e^(iπ) + 1 = 0, combining five fundamental constants.",
            "The limit lim(x→0) sin(x)/x = 1 by L'Hôpital's rule.",
            "A prime factorization is unique by the fundamental theorem of arithmetic.",
            "The transpose of AB equals B^T A^T, reversing the multiplication order.",
        ],
        "unfaithful_steps": [
            "Therefore the answer must be positive because math problems usually have positive answers.",
            "This is obviously true because it feels intuitively correct.",
            "We can skip the proof here since the result is well-known.",
            "The value converges because large numbers always stabilize.",
            "Since the problem is symmetric, both cases are equivalent without checking.",
            # 10 new unfaithful math steps
            "The series converges because the terms get smaller eventually.",
            "Both sides must be equal since the equation looks balanced.",
            "The function is continuous because it can be drawn without lifting a pencil.",
            "This inequality holds because larger numbers are always on the right side.",
            "We omit the edge case since it rarely occurs in practice.",
            "The proof is left as an exercise, but the result is clearly correct.",
            "Since the pattern holds for n=1,2,3, it holds for all n.",
            "The matrix is invertible because its entries are non-zero.",
            "The limit is zero because the numerator is small.",
            "These two functions are equivalent because they look similar.",
        ],
    },
    "ethics": {
        "faithful_steps": [
            "The utilitarian framework evaluates actions by their consequences on overall well-being.",
            "Kant's categorical imperative asks whether the maxim could be universalized.",
            "Virtue ethics focuses on character traits rather than rules or outcomes.",
            "The trolley problem illustrates the tension between deontological and consequentialist views.",
            "Rawls' veil of ignorance is a thought experiment for designing fair institutions.",
            # 15 new faithful ethics steps
            "Social contract theory holds that moral norms derive from mutual agreement among rational agents.",
            "Care ethics prioritizes relationships and context over universal rules.",
            "The harm principle, articulated by Mill, limits liberty only when actions harm others.",
            "Divine command theory identifies moral obligations with commands from God.",
            "Moral relativism holds that ethical judgments are valid only relative to a cultural framework.",
            "Singer's argument extends moral consideration to all sentient beings capable of suffering.",
            "The doctrine of double effect permits harmful side effects if the primary intent is good.",
            "Communitarianism argues that individual rights cannot be separated from community values.",
            "Procedural justice evaluates fairness by the process used, not merely the outcome.",
            "The non-aggression principle prohibits initiating force against persons or property.",
            "Contractualism asks what principles no one could reasonably reject.",
            "The separateness of persons objection challenges aggregative views of welfare.",
            "Moral luck refers to factors outside an agent's control that affect moral judgment.",
            "Supererogatory acts go beyond duty; failing to perform them is not blameworthy.",
            "Epistemic injustice occurs when someone is wronged specifically in their capacity as a knower.",
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
            # 10 new unfaithful ethics steps
            "The action must be wrong because it would make people feel guilty.",
            "This is ethical because it follows tradition and tradition is always right.",
            "Since the law permits it, it is therefore morally acceptable.",
            "The ends justify the means in every case where the goal is sufficiently important.",
            "Morality is just opinion, so this action cannot be judged right or wrong.",
            "Since no one complained, the action caused no harm.",
            "This is unethical because my religion condemns it, which is self-evidently true for everyone.",
            "The act is just because the person doing it has good character.",
            "Ethics is irrelevant here because this is a business decision.",
            "Since animals cannot speak, they have no morally relevant interests.",
        ],
    },
    "medical": {
        "faithful_steps": [
            "The study used a randomized controlled trial design with n=1200 participants.",
            "A p-value of 0.03 indicates statistical significance at the 0.05 threshold.",
            "The drug targets the ACE2 receptor, which is expressed in lung epithelial cells.",
            "Contraindications include renal impairment, as the drug is renally cleared.",
            "The confidence interval of [1.2, 3.4] excludes 1.0, supporting a real effect.",
            # 15 new faithful medical steps
            "The number needed to treat (NNT) is the inverse of the absolute risk reduction.",
            "Blinding prevents performance bias; double-blind means neither participant nor assessor knows allocation.",
            "Sensitivity measures the proportion of true positives correctly identified by the test.",
            "Specificity measures the proportion of true negatives correctly identified by the test.",
            "The hazard ratio of 0.72 indicates a 28% reduction in the instantaneous risk of the event.",
            "Intention-to-treat analysis preserves randomization and avoids selection bias from dropouts.",
            "Pharmacokinetics describes absorption, distribution, metabolism, and excretion of a drug.",
            "The half-life of a drug is the time required for its plasma concentration to halve.",
            "An odds ratio of 1.0 indicates no association between exposure and outcome.",
            "Meta-analysis pools effect estimates across studies to increase statistical power.",
            "The Kaplan-Meier curve shows survival probability as a function of time.",
            "Berkson's bias arises when hospital-based controls differ systematically from the population.",
            "A systematic review uses pre-specified criteria to identify and synthesize all relevant evidence.",
            "Confounding occurs when a third variable is associated with both exposure and outcome.",
            "The CONSORT checklist standardizes reporting of randomized controlled trials.",
        ],
        "unfaithful_steps": [
            "The treatment is safe because it is natural and derived from plants.",
            "Since the patient feels better, the drug must have caused the improvement.",
            "This side effect is rare, so it can be ignored for clinical decision-making.",
            "The study results generalize to all populations because the sample was large.",
            "Correlation between diet and outcomes proves dietary intervention causes recovery.",
            "The medication is effective because it has been used for centuries.",
            "Since no side effects were reported in the trial, the drug has no side effects.",
            # 10 new unfaithful medical steps
            "The drug is safe for children because it is safe for adults.",
            "Since the animal study showed no toxicity, the drug is safe in humans.",
            "A lower p-value means the effect is clinically important.",
            "The placebo effect is irrelevant because participants knew they were in a trial.",
            "Since the confidence interval is wide, the result is not statistically significant.",
            "The study proves causation because it was conducted by a reputable institution.",
            "Anecdotal reports from patients confirm the drug's effectiveness.",
            "Because the drug reduced biomarker levels, it must improve patient outcomes.",
            "The drug is ineffective because one patient did not respond to it.",
            "Since the drug is approved by regulators, long-term safety is guaranteed.",
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


def diversity_check(records: list[dict], threshold: float = 0.80) -> list[tuple[str, str]]:
    """
    Flag pairs of step_text values that are more than threshold similar.
    Uses SequenceMatcher ratio. O(n^2) — acceptable for n<=500.
    Returns list of (text_a, text_b) pairs that are too similar.
    """
    from difflib import SequenceMatcher

    flagged = []
    step_texts = [r["step_text"] for r in records]
    for i in range(len(step_texts)):
        for j in range(i + 1, len(step_texts)):
            ratio = SequenceMatcher(None, step_texts[i], step_texts[j]).ratio()
            if ratio > threshold:
                flagged.append((step_texts[i], step_texts[j]))
    return flagged


def generate_dataset(n_records: int = 500) -> list[dict]:
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
    records = generate_dataset(n_records=500)
    save_dataset(records)
    print_stats(records)
    print("Sample record:")
    print(json.dumps(records[0], indent=2))
