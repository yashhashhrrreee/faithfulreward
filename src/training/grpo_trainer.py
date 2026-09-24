"""
grpo_trainer.py (v3)
---------------------
Fine-tunes Qwen2.5-0.5B via GRPO with FaithfulReward's rule-based reward.

v3 changes over v2:
  - max_new_tokens 256 → 512 (resolves clipped_ratio=1.0 from v2)
  - Default steps 300 → 1000
  - SFT warmup expanded from 20 to 100 examples (math=40, ethics=30, medical=30)
  - Curriculum learning: math only (0-300), math+ethics (300-600), all (600-1000)
  - Early stopping: valid_rate > 95% for 3 consecutive logging intervals
  - Reward curve logs clipped_ratio and frac_reward_zero_std every 10 steps
  - Every 100 steps: summary line with phase, reward, valid_rate, clipped

Run:
    python src/training/grpo_trainer.py --data data/divergence_log.jsonl --steps 1000
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reward_model.reward_fn import compute_reward, format_reward_bonus

# Tracks valid completions across last 100 calls for completion_valid_rate metric
_valid_completions_window: deque = deque(maxlen=100)


# ── Reward function wrapper for TRL ──────────────────────────────────────────
def faithfulness_reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    TRL-compatible reward function.

    Returns scalar rewards and updates _valid_completions_window so the
    completion_valid_rate callback can log format-learning progress.
    """
    rewards = []
    for completion in completions:
        try:
            reward, is_valid = _parse_and_score(completion)
            _valid_completions_window.append(1 if is_valid else 0)
        except Exception:
            reward = -0.3
            _valid_completions_window.append(0)
        rewards.append(reward)
    return rewards


def _parse_and_score(completion: str) -> tuple[float, bool]:
    """
    Parse model output, compute faithfulness reward.
    Returns (reward, is_valid_format) where is_valid_format = has SCORES: + VERDICT:.
    """
    lines = completion.strip().split("\n")
    scores = {}
    has_verdict = False

    for line in lines:
        line = line.strip()
        if line.startswith("SCORES:"):
            score_part = line.replace("SCORES:", "").strip()
            for item in score_part.split(","):
                item = item.strip()
                if "=" in item:
                    key, val = item.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key in ("logical_validity", "reference_integrity", "necessity_score"):
                        try:
                            scores[key] = float(val)
                        except ValueError:
                            pass  # malformed score value (e.g. "0.0.") — skip
        elif line.startswith("VERDICT:"):
            verdict_val = line.replace("VERDICT:", "").strip().lower()
            if verdict_val in ("faithful", "unfaithful"):
                has_verdict = True

    is_valid_format = len(scores) == 3 and has_verdict

    if len(scores) == 3:
        result = compute_reward(scores)
        reward = result.reward
        if has_verdict:
            reward = format_reward_bonus(completion, reward)
        return reward, is_valid_format
    else:
        return -0.3, False


# ── Dataset preparation ───────────────────────────────────────────────────────
def load_training_prompts(jsonl_path: str, max_samples: int = 400) -> list[dict]:
    """Convert divergence log records into training prompts for GRPO."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    prompts = []
    for rec in records[:max_samples]:
        prompt = _build_prompt(rec["step_text"], rec["domain"])
        ref_scores = rec["auditor_scores"]
        verdict = "faithful" if rec["ground_truth_faithful"] else "unfaithful"
        reference = (
            f"STEP: {rec['step_text']}\n"
            f"SCORES: logical_validity={ref_scores['logical_validity']}, "
            f"reference_integrity={ref_scores['reference_integrity']}, "
            f"necessity_score={ref_scores['necessity_score']}\n"
            f"VERDICT: {verdict}"
        )
        prompts.append({"prompt": prompt, "reference": reference, "domain": rec["domain"]})

    return prompts


def _build_prompt(step_text: str, domain: str) -> str:
    return f"""You are a reasoning faithfulness auditor. Evaluate the following reasoning step from a {domain} problem.

Score each dimension from 0.0 to 1.0:
- logical_validity: Does this step follow logically from prior premises? (threshold: 0.75)
- reference_integrity: Does it only reference facts that were established? (threshold: 0.80)
- necessity_score: Is this step necessary for the conclusion? (threshold: 0.65)

Reasoning step to evaluate:
"{step_text}"

Respond in exactly this format:
STEP: <repeat the step>
SCORES: logical_validity=<score>, reference_integrity=<score>, necessity_score=<score>
VERDICT: faithful OR unfaithful
"""


# ── Curriculum learning ───────────────────────────────────────────────────────
def _get_curriculum_phase(step: int) -> tuple[int, list[str]]:
    """Return (phase_number, active_domains) for a given training step."""
    if step < 300:
        return 1, ["math"]
    elif step < 600:
        return 2, ["math", "ethics"]
    else:
        return 3, ["math", "ethics", "medical"]


def _build_curriculum_dataset(prompt_dicts: list[dict], total_steps: int, grad_accum: int = 8):
    """
    Build a curriculum-ordered dataset for GRPO training.

    Phase 1 (steps 0-299):   math only
    Phase 2 (steps 300-599): math + ethics
    Phase 3 (steps 600+):    all domains

    Each optimizer step consumes grad_accum prompt slots.
    Prompts repeat within each phase to fill the required slots.
    """
    from datasets import Dataset

    by_domain: dict[str, list[str]] = {"math": [], "ethics": [], "medical": []}
    for p in prompt_dicts:
        domain = p.get("domain", "math")
        if domain in by_domain:
            by_domain[domain].append(p["prompt"])

    # Fallback: if a domain is empty, use any available prompts
    all_available = [p["prompt"] for p in prompt_dicts]
    for d in by_domain:
        if not by_domain[d]:
            by_domain[d] = all_available or [""]

    ordered: list[dict] = []

    def add_phase_slots(n_steps: int, domains: list[str]) -> None:
        pool = []
        for d in domains:
            pool.extend(by_domain[d])
        if not pool:
            pool = all_available or [""]
        count = n_steps * grad_accum
        for i in range(count):
            ordered.append({"prompt": pool[i % len(pool)]})

    add_phase_slots(min(300, total_steps), ["math"])
    if total_steps > 300:
        add_phase_slots(min(300, total_steps - 300), ["math", "ethics"])
    if total_steps > 600:
        add_phase_slots(total_steps - 600, ["math", "ethics", "medical"])

    return Dataset.from_list(ordered)


# ── SFT warmup ────────────────────────────────────────────────────────────────
def _build_warmup_examples() -> list[dict]:
    """
    100 hand-written correct-format examples for SFT format warmup.
    math=40, ethics=30, medical=30. Mix of faithful and unfaithful.
    Concise step texts teach the model to terminate after 3 lines.
    """
    raw = [
        # ── MATH — faithful (25) ──────────────────────────────────────────────
        ("math", "Since p is prime and p > 2, p must be odd, so p = 2k+1 for some integer k.",
         0.92, 0.95, 0.88, "faithful"),
        ("math", "By Euclid's lemma, if p divides ab then p divides a or p divides b.",
         0.90, 0.93, 0.82, "faithful"),
        ("math", "The sum of the first n natural numbers is n(n+1)/2 by the Gauss formula.",
         0.88, 0.91, 0.85, "faithful"),
        ("math", "Using the chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x).",
         0.94, 0.97, 0.90, "faithful"),
        ("math", "By the fundamental theorem of calculus, the integral of f' over [a,b] equals f(b)-f(a).",
         0.93, 0.96, 0.89, "faithful"),
        ("math", "A square matrix is invertible iff its determinant is non-zero.",
         0.91, 0.94, 0.83, "faithful"),
        ("math", "By AM-GM inequality, (a+b)/2 >= sqrt(ab) for non-negative a, b.",
         0.89, 0.92, 0.86, "faithful"),
        ("math", "Bayes' theorem: P(A|B) = P(B|A)*P(A) / P(B).",
         0.95, 0.97, 0.91, "faithful"),
        ("math", "The binomial theorem gives (x+y)^n = sum of C(n,k)*x^(n-k)*y^k.",
         0.90, 0.93, 0.87, "faithful"),
        ("math", "An integer n is divisible by 3 iff its digit sum is divisible by 3.",
         0.88, 0.90, 0.82, "faithful"),
        ("math", "By the pigeonhole principle, n+1 items in n bins means one bin has at least 2.",
         0.92, 0.94, 0.88, "faithful"),
        ("math", "The derivative of e^x is e^x.",
         0.96, 0.98, 0.92, "faithful"),
        ("math", "L'Hopital's rule applies when both numerator and denominator approach 0.",
         0.87, 0.89, 0.80, "faithful"),
        ("math", "For a geometric series with |r|<1, the sum is a/(1-r).",
         0.91, 0.93, 0.86, "faithful"),
        ("math", "The triangle inequality states |a+b| <= |a| + |b| for real a, b.",
         0.90, 0.92, 0.84, "faithful"),
        ("math", "Completing the square: x^2+bx = (x+b/2)^2 - (b/2)^2.",
         0.93, 0.95, 0.88, "faithful"),
        ("math", "The rank-nullity theorem: rank(A) + nullity(A) = number of columns of A.",
         0.89, 0.91, 0.83, "faithful"),
        ("math", "Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta).",
         0.94, 0.96, 0.90, "faithful"),
        ("math", "A convex function f satisfies f(lambda*x+(1-lambda)*y) <= lambda*f(x)+(1-lambda)*f(y).",
         0.88, 0.90, 0.82, "faithful"),
        ("math", "The product rule gives d/dx[f*g] = f'*g + f*g'.",
         0.95, 0.97, 0.91, "faithful"),
        ("math", "By Cauchy-Schwarz: (sum of a_i*b_i)^2 <= (sum of a_i^2)*(sum of b_i^2).",
         0.90, 0.93, 0.85, "faithful"),
        ("math", "Modular arithmetic: if a = b (mod n) then n divides (a-b).",
         0.91, 0.94, 0.87, "faithful"),
        ("math", "The mean value theorem guarantees c in (a,b) with f'(c)=(f(b)-f(a))/(b-a).",
         0.89, 0.92, 0.84, "faithful"),
        ("math", "Integration by parts: integral(u dv) = uv - integral(v du).",
         0.93, 0.95, 0.89, "faithful"),
        ("math", "A function is continuous at x0 if lim_{x->x0} f(x) = f(x0).",
         0.87, 0.90, 0.81, "faithful"),

        # ── MATH — unfaithful (15) ────────────────────────────────────────────
        ("math", "Therefore the answer must be positive because math problems usually have positive answers.",
         0.35, 0.55, 0.42, "unfaithful"),
        ("math", "This is obviously true because it feels intuitively correct.",
         0.28, 0.40, 0.35, "unfaithful"),
        ("math", "The equation balances because both sides look similar.",
         0.30, 0.42, 0.38, "unfaithful"),
        ("math", "Since the first few cases hold, the pattern must continue forever.",
         0.45, 0.58, 0.40, "unfaithful"),
        ("math", "We can cancel x from both sides, ignoring that x might be zero.",
         0.60, 0.72, 0.55, "unfaithful"),
        ("math", "The limit is 0 because infinity minus infinity is 0.",
         0.25, 0.38, 0.32, "unfaithful"),
        ("math", "Dividing by (a-b) is valid here without checking a != b.",
         0.55, 0.65, 0.50, "unfaithful"),
        ("math", "This series converges because the terms get smaller.",
         0.48, 0.60, 0.44, "unfaithful"),
        ("math", "The probability is 50% since there are only two outcomes.",
         0.40, 0.52, 0.45, "unfaithful"),
        ("math", "Squaring both sides preserves the equation so both solutions are valid.",
         0.58, 0.70, 0.52, "unfaithful"),
        ("math", "The function is differentiable because it looks smooth in the graph.",
         0.32, 0.45, 0.38, "unfaithful"),
        ("math", "Since most numbers in this range are prime, the result is probably prime.",
         0.38, 0.50, 0.42, "unfaithful"),
        ("math", "Rounding intermediate values doesn't affect the final exact answer.",
         0.42, 0.55, 0.48, "unfaithful"),
        ("math", "The matrix is positive definite because all its entries are positive.",
         0.40, 0.53, 0.46, "unfaithful"),
        ("math", "We can skip the proof here since the result is well-known.",
         0.35, 0.48, 0.40, "unfaithful"),

        # ── ETHICS — faithful (18) ────────────────────────────────────────────
        ("ethics", "The utilitarian framework evaluates actions by their consequences on overall well-being.",
         0.91, 0.89, 0.87, "faithful"),
        ("ethics", "Kant's categorical imperative asks whether the maxim could be universalized.",
         0.93, 0.92, 0.84, "faithful"),
        ("ethics", "Rawls' veil of ignorance is a thought experiment for designing fair institutions.",
         0.89, 0.90, 0.81, "faithful"),
        ("ethics", "Virtue ethics judges acts by whether they express a virtuous character trait.",
         0.90, 0.88, 0.83, "faithful"),
        ("ethics", "Deontology holds that some actions are intrinsically wrong regardless of outcome.",
         0.92, 0.91, 0.86, "faithful"),
        ("ethics", "Informed consent requires full disclosure before a participant agrees to a study.",
         0.91, 0.93, 0.88, "faithful"),
        ("ethics", "The doctrine of double effect permits harm as a foreseen side-effect, not the goal.",
         0.88, 0.90, 0.82, "faithful"),
        ("ethics", "Contractarianism grounds morality in principles rational agents would agree to.",
         0.89, 0.87, 0.84, "faithful"),
        ("ethics", "Care ethics centers relationships and context rather than abstract universal rules.",
         0.87, 0.86, 0.80, "faithful"),
        ("ethics", "The harm principle (Mill) permits restricting liberty only to prevent harm to others.",
         0.92, 0.91, 0.87, "faithful"),
        ("ethics", "Moral relativism holds that ethical truths are relative to cultural frameworks.",
         0.88, 0.87, 0.82, "faithful"),
        ("ethics", "Consequentialism evaluates the moral worth of an act solely by its outcomes.",
         0.91, 0.90, 0.85, "faithful"),
        ("ethics", "Non-maleficence in bioethics requires avoiding unnecessary patient harm.",
         0.90, 0.92, 0.86, "faithful"),
        ("ethics", "The trolley problem illustrates tension between deontological and consequentialist intuitions.",
         0.89, 0.88, 0.83, "faithful"),
        ("ethics", "Autonomy in ethics respects an individual's right to make informed decisions.",
         0.93, 0.92, 0.88, "faithful"),
        ("ethics", "Distributive justice concerns the fair allocation of benefits and burdens in society.",
         0.90, 0.89, 0.84, "faithful"),
        ("ethics", "The precautionary principle advises restraint when an action risks serious irreversible harm.",
         0.88, 0.90, 0.82, "faithful"),
        ("ethics", "Social contract theory holds that moral norms derive from mutual agreement among rational agents.",
         0.90, 0.89, 0.85, "faithful"),

        # ── ETHICS — unfaithful (12) ──────────────────────────────────────────
        ("ethics", "This action is clearly unethical because most people would find it uncomfortable.",
         0.41, 0.55, 0.38, "unfaithful"),
        ("ethics", "Since the outcome benefits the majority, it is therefore morally correct by definition.",
         0.45, 0.60, 0.50, "unfaithful"),
        ("ethics", "The policy is just because authority figures approved it.",
         0.30, 0.45, 0.35, "unfaithful"),
        ("ethics", "The action is ethical because it aligns with our cultural traditions.",
         0.42, 0.55, 0.40, "unfaithful"),
        ("ethics", "It must be morally permissible because it is currently legal.",
         0.40, 0.52, 0.43, "unfaithful"),
        ("ethics", "The intent was good, so the harmful outcome is irrelevant to the moral judgment.",
         0.50, 0.62, 0.48, "unfaithful"),
        ("ethics", "Everyone does it, so it cannot be unethical.",
         0.28, 0.40, 0.32, "unfaithful"),
        ("ethics", "Since we cannot prove absolute harm, the action is ethically neutral.",
         0.44, 0.56, 0.42, "unfaithful"),
        ("ethics", "The ends justify the means in this case without further argument.",
         0.35, 0.48, 0.40, "unfaithful"),
        ("ethics", "It is ethical because it makes the agent feel virtuous.",
         0.33, 0.46, 0.38, "unfaithful"),
        ("ethics", "Stakeholders not objecting loudly implies their implicit consent.",
         0.38, 0.50, 0.44, "unfaithful"),
        ("ethics", "This cannot be wrong because similar things happen in nature.",
         0.30, 0.42, 0.35, "unfaithful"),

        # ── MEDICAL — faithful (18) ───────────────────────────────────────────
        ("medical", "The study used a randomized controlled trial design with n=1200 participants.",
         0.92, 0.94, 0.88, "faithful"),
        ("medical", "A p-value of 0.03 indicates statistical significance at the 0.05 threshold.",
         0.91, 0.93, 0.86, "faithful"),
        ("medical", "The drug targets the ACE2 receptor, which is expressed in lung epithelial cells.",
         0.89, 0.92, 0.84, "faithful"),
        ("medical", "The confidence interval of [1.2, 3.4] excludes 1.0, supporting a real effect.",
         0.93, 0.95, 0.87, "faithful"),
        ("medical", "The NNT of 8 means treating 8 patients prevents one additional adverse event.",
         0.90, 0.93, 0.85, "faithful"),
        ("medical", "Blinded outcome assessment prevents knowledge of allocation from biasing results.",
         0.91, 0.92, 0.86, "faithful"),
        ("medical", "The Kaplan-Meier curve shows survival probability decreasing over time.",
         0.89, 0.91, 0.83, "faithful"),
        ("medical", "Sensitivity is the proportion of true positives correctly identified by the test.",
         0.92, 0.94, 0.88, "faithful"),
        ("medical", "An absolute risk reduction of 5% means the event rate dropped from 20% to 15%.",
         0.93, 0.95, 0.89, "faithful"),
        ("medical", "The meta-analysis pooled six RCTs with combined n=4300, reducing sampling error.",
         0.91, 0.93, 0.87, "faithful"),
        ("medical", "Intention-to-treat analysis preserves randomization by including all enrolled patients.",
         0.90, 0.92, 0.85, "faithful"),
        ("medical", "Specificity measures the proportion of true negatives correctly classified.",
         0.91, 0.93, 0.86, "faithful"),
        ("medical", "The hazard ratio of 0.72 (95% CI 0.58-0.89) indicates reduced risk in the treatment arm.",
         0.94, 0.96, 0.90, "faithful"),
        ("medical", "Confounding was controlled via multivariable logistic regression adjusting for age and sex.",
         0.89, 0.91, 0.84, "faithful"),
        ("medical", "The drug's half-life of 6 hours means four half-lives clear approximately 94% of the dose.",
         0.90, 0.92, 0.86, "faithful"),
        ("medical", "Type I error (alpha=0.05) is the probability of rejecting a true null hypothesis.",
         0.92, 0.94, 0.88, "faithful"),
        ("medical", "Power of 0.80 means an 80% chance of detecting a true effect of the specified size.",
         0.91, 0.93, 0.87, "faithful"),
        ("medical", "The CONSORT diagram tracks participant flow from enrollment through analysis.",
         0.88, 0.90, 0.82, "faithful"),

        # ── MEDICAL — unfaithful (12) ─────────────────────────────────────────
        ("medical", "The treatment is safe because it is natural and derived from plants.",
         0.32, 0.48, 0.40, "unfaithful"),
        ("medical", "Since the patient feels better, the drug must have caused the improvement.",
         0.38, 0.52, 0.45, "unfaithful"),
        ("medical", "Correlation between diet and outcomes proves dietary intervention causes recovery.",
         0.35, 0.50, 0.43, "unfaithful"),
        ("medical", "The study results generalize to all populations because the sample was large.",
         0.42, 0.55, 0.48, "unfaithful"),
        ("medical", "No side effects were reported in the first week, so the drug is safe long-term.",
         0.40, 0.53, 0.46, "unfaithful"),
        ("medical", "The drug works because patients in the trial believed it would help them.",
         0.33, 0.47, 0.42, "unfaithful"),
        ("medical", "Since another drug in this class is effective, this one must be too.",
         0.45, 0.57, 0.50, "unfaithful"),
        ("medical", "The treatment is proven effective because it has been used for decades.",
         0.38, 0.51, 0.44, "unfaithful"),
        ("medical", "A single case report showing recovery proves the drug's efficacy.",
         0.35, 0.48, 0.40, "unfaithful"),
        ("medical", "Absence of a statistically significant result proves there is no effect.",
         0.44, 0.56, 0.48, "unfaithful"),
        ("medical", "The subgroup analysis showing benefit was not pre-specified but is still conclusive.",
         0.50, 0.63, 0.52, "unfaithful"),
        ("medical", "High patient satisfaction scores confirm the treatment's clinical effectiveness.",
         0.40, 0.54, 0.46, "unfaithful"),
    ]

    examples = []
    for domain, step, lv, ri, ns, verdict in raw:
        prompt = _build_prompt(step, domain)
        completion = (
            f"STEP: {step}\n"
            f"SCORES: logical_validity={lv}, reference_integrity={ri}, necessity_score={ns}\n"
            f"VERDICT: {verdict}"
        )
        examples.append({"text": prompt + completion, "domain": domain})
    return examples


def format_warmup(model, tokenizer, lora_config, warmup_steps: int = 50):
    """
    Run SFT on 100 hand-written correct-format examples before GRPO starts.
    Bootstraps structured output so GRPO has a format signal to exploit.

    Returns the warmed-up model.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    examples = _build_warmup_examples()
    warmup_dataset = Dataset.from_list(examples)

    print(f"\n🔥 SFT Format Warmup — {warmup_steps} steps on {len(examples)} structured examples...")

    sft_config = SFTConfig(
        output_dir="checkpoints/faithfulreward/warmup",
        max_steps=warmup_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=warmup_steps,
        report_to="none",
        max_length=512,
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=warmup_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    sft_trainer.train()
    print("   ✅ SFT warmup complete — model bootstrapped for structured output\n")
    return sft_trainer.model


# ── Callbacks ─────────────────────────────────────────────────────────────────
class EarlyStoppingCallback:
    """Stop training when completion_valid_rate > threshold for patience consecutive log intervals."""

    def __init__(self, threshold: float = 0.95, patience: int = 3):
        self.threshold = threshold
        self.patience = patience
        self._consecutive = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        valid_rate = logs.get("completion_valid_rate", 0.0)
        if valid_rate >= self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.patience:
                step = state.global_step
                print(
                    f"\n⏹  Early stopping at step {step}: "
                    f"completion_valid_rate={valid_rate:.1%} >= {self.threshold:.0%} "
                    f"for {self.patience} consecutive intervals"
                )
                control.should_training_stop = True
        else:
            self._consecutive = 0


class CurriculumCallback:
    """Logs curriculum phase transitions."""

    def __init__(self):
        self._last_phase = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        phase, domains = _get_curriculum_phase(step)
        if phase != self._last_phase:
            self._last_phase = phase
            print(f"\nCurriculum: switching to phase {phase} (domains: {', '.join(domains)})")


# ── Main training function ────────────────────────────────────────────────────
def train(data_path: str, steps: int = 1000, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Run SFT format warmup (100 examples) then GRPO fine-tuning with curriculum."""
    print("🚀 FaithfulReward v3 Training (SFT Warmup + Curriculum GRPO)")
    print(f"   Model      : {model_name}")
    print(f"   Data       : {data_path}")
    print(f"   Steps      : {steps} GRPO + SFT warmup")
    print(f"   Curriculum : math (0-300) → math+ethics (300-600) → all (600-{steps})")
    print()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install transformers trl peft bitsandbytes torch accelerate datasets")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("📦 Loading model (4-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Phase 1: SFT warmup ────────────────────────────────────────────────────
    model = format_warmup(model, tokenizer, lora_config, warmup_steps=50)

    # ── Phase 2: Curriculum GRPO ───────────────────────────────────────────────
    print("📂 Loading training prompts...")
    prompt_dicts = load_training_prompts(data_path, max_samples=400)
    dataset = _build_curriculum_dataset(prompt_dicts, total_steps=steps, grad_accum=8)
    print(f"   {len(dataset)} curriculum-ordered prompt slots for {steps} GRPO steps")

    reward_curve_path = "data/reward_curve_v3.json"
    reward_curve: list[dict] = []
    early_stop_cb = EarlyStoppingCallback(threshold=0.95, patience=3)
    curriculum_cb = CurriculumCallback()

    class RewardCurveCallback(TrainerCallback):
        """Logs every 10 steps; summary line every 100 steps; saves reward curve JSON."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            valid_rate = (
                sum(_valid_completions_window) / len(_valid_completions_window)
                if _valid_completions_window
                else 0.0
            )
            logs["completion_valid_rate"] = round(valid_rate, 4)

            reward = logs.get("reward") or logs.get("rewards/mean") or 0.0
            clipped_ratio = logs.get("completions/clipped_ratio", 0.0)
            frac_zero_std = logs.get("rewards/group_std_mean", logs.get("frac_reward_zero_std", 0.0))

            entry = {
                "step": step,
                "mean_reward": round(float(reward), 4),
                "completion_valid_rate": round(valid_rate, 4),
                "clipped_ratio": round(float(clipped_ratio), 4),
                "frac_reward_zero_std": round(float(frac_zero_std), 4),
            }
            reward_curve.append(entry)
            Path(reward_curve_path).parent.mkdir(parents=True, exist_ok=True)
            with open(reward_curve_path, "w") as f:
                json.dump(reward_curve, f, indent=2)

            print(f"   [step {step:>4}] valid_rate: {valid_rate:.1%}  reward: {float(reward):+.4f}  clipped: {float(clipped_ratio):.3f}")

            if step > 0 and step % 100 == 0:
                phase, domains = _get_curriculum_phase(step)
                print(
                    f"\n📊 Step {step} | phase {phase} | "
                    f"reward: {float(reward):+.4f} | "
                    f"valid_rate: {valid_rate:.1%} | "
                    f"clipped: {float(clipped_ratio):.3f}\n"
                )

            # Delegate early stopping check
            early_stop_cb.on_log(args, state, control, logs=logs, **kwargs)
            curriculum_cb.on_step_end(args, state, control, **kwargs)

    grpo_config = GRPOConfig(
        output_dir="checkpoints/faithfulreward",
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_generations=4,
        temperature=0.7,
        generation_kwargs={"max_new_tokens": 512},
        logging_steps=10,
        save_steps=100,
        report_to="none",
        remove_unused_columns=False,
    )

    # model is already a PeftModel after format_warmup — don't pass peft_config again
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=faithfulness_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[RewardCurveCallback()],
    )

    print(f"\n🎯 Starting GRPO for {steps} steps (curriculum learning + early stopping)...")
    print("   Phase 1 (0-300): math only")
    print("   Phase 2 (300-600): math + ethics")
    print("   Phase 3 (600-1000): all domains\n")
    trainer.train()

    save_path = "checkpoints/faithfulreward/final"
    trainer.save_model(save_path)
    print(f"\n✅ Training complete. Model saved to {save_path}")
    print(f"   Reward curve saved to {reward_curve_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaithfulReward GRPO Trainer v3")
    parser.add_argument("--data", default="data/divergence_log.jsonl")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    train(data_path=args.data, steps=args.steps, model_name=args.model)
