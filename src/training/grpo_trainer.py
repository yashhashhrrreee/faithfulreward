"""
grpo_trainer.py (v2)
---------------------
Fine-tunes Qwen2.5-0.5B via GRPO with FaithfulReward's rule-based reward.

v2 changes over v1:
  - max_new_tokens 100 → 256 (fixes truncation before SCORES: line)
  - Default steps 100 → 300
  - format_warmup(): 50 SFT steps on 20 hand-written examples before GRPO
  - completion_valid_rate metric logged each step
  - Format reward bonus (+0.1) for valid SCORES: + VERDICT: output

Run:
    python src/training/grpo_trainer.py --data data/divergence_log.jsonl --steps 300
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
                        scores[key] = float(val)
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
def load_training_prompts(jsonl_path: str, max_samples: int = 200) -> list[dict]:
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


# ── SFT warmup ────────────────────────────────────────────────────────────────
def _build_warmup_examples() -> list[dict]:
    """
    20 hand-written correct-format examples for SFT format warmup.
    Each shows exactly what a valid SCORES:/VERDICT: completion looks like.
    """
    raw = [
        # Math — faithful
        ("math", "Since p is prime and p > 2, p must be odd, so p = 2k+1 for some integer k.",
         0.92, 0.95, 0.88, "faithful"),
        ("math", "By Euclid's lemma, if p divides ab then p divides a or p divides b.",
         0.90, 0.93, 0.82, "faithful"),
        ("math", "The sum of the first n natural numbers is n(n+1)/2 by the Gauss formula.",
         0.88, 0.91, 0.85, "faithful"),
        ("math", "Using the chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x).",
         0.94, 0.97, 0.90, "faithful"),
        # Math — unfaithful
        ("math", "Therefore the answer must be positive because math problems usually have positive answers.",
         0.35, 0.55, 0.42, "unfaithful"),
        ("math", "This is obviously true because it feels intuitively correct.",
         0.28, 0.40, 0.35, "unfaithful"),
        # Ethics — faithful
        ("ethics", "The utilitarian framework evaluates actions by their consequences on overall well-being.",
         0.91, 0.89, 0.87, "faithful"),
        ("ethics", "Kant's categorical imperative asks whether the maxim could be universalized.",
         0.93, 0.92, 0.84, "faithful"),
        ("ethics", "Rawls' veil of ignorance is a thought experiment for designing fair institutions.",
         0.89, 0.90, 0.81, "faithful"),
        # Ethics — unfaithful
        ("ethics", "This action is clearly unethical because most people would find it uncomfortable.",
         0.41, 0.55, 0.38, "unfaithful"),
        ("ethics", "Since the outcome benefits the majority, it is therefore morally correct by definition.",
         0.45, 0.60, 0.50, "unfaithful"),
        ("ethics", "The policy is just because authority figures approved it.",
         0.30, 0.45, 0.35, "unfaithful"),
        # Medical — faithful
        ("medical", "The study used a randomized controlled trial design with n=1200 participants.",
         0.92, 0.94, 0.88, "faithful"),
        ("medical", "A p-value of 0.03 indicates statistical significance at the 0.05 threshold.",
         0.91, 0.93, 0.86, "faithful"),
        ("medical", "The drug targets the ACE2 receptor, which is expressed in lung epithelial cells.",
         0.89, 0.92, 0.84, "faithful"),
        ("medical", "The confidence interval of [1.2, 3.4] excludes 1.0, supporting a real effect.",
         0.93, 0.95, 0.87, "faithful"),
        # Medical — unfaithful
        ("medical", "The treatment is safe because it is natural and derived from plants.",
         0.32, 0.48, 0.40, "unfaithful"),
        ("medical", "Since the patient feels better, the drug must have caused the improvement.",
         0.38, 0.52, 0.45, "unfaithful"),
        ("medical", "Correlation between diet and outcomes proves dietary intervention causes recovery.",
         0.35, 0.50, 0.43, "unfaithful"),
        ("medical", "The study results generalize to all populations because the sample was large.",
         0.42, 0.55, 0.48, "unfaithful"),
    ]

    examples = []
    for domain, step, lv, ri, ns, verdict in raw:
        prompt = _build_prompt(step, domain)
        completion = (
            f"STEP: {step}\n"
            f"SCORES: logical_validity={lv}, reference_integrity={ri}, necessity_score={ns}\n"
            f"VERDICT: {verdict}"
        )
        examples.append({"text": prompt + completion})
    return examples


def format_warmup(model, tokenizer, lora_config, warmup_steps: int = 50):
    """
    Run SFT on 20 hand-written correct-format examples before GRPO starts.
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


# ── Main training function ────────────────────────────────────────────────────
def train(data_path: str, steps: int = 300, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Run SFT format warmup (50 steps) then GRPO fine-tuning (steps)."""
    print("🚀 FaithfulReward v2 Training (SFT Warmup + GRPO)")
    print(f"   Model      : {model_name}")
    print(f"   Data       : {data_path}")
    print(f"   Steps      : {steps} GRPO + 50 SFT warmup")
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

    # ── Phase 2: GRPO ─────────────────────────────────────────────────────────
    print("📂 Loading training prompts...")
    prompt_dicts = load_training_prompts(data_path, max_samples=200)
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in prompt_dicts])
    print(f"   {len(dataset)} training examples loaded")

    reward_curve_path = "data/reward_curve.json"
    reward_curve: list[dict] = []

    class RewardCurveCallback(TrainerCallback):
        """Logs completion_valid_rate each step and saves reward curve to JSON."""

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

            reward = logs.get("reward") or logs.get("rewards/mean")
            if reward is not None:
                entry = {
                    "step": step,
                    "mean_reward": round(float(reward), 4),
                    "completion_valid_rate": round(valid_rate, 4),
                }
                reward_curve.append(entry)
                Path(reward_curve_path).parent.mkdir(parents=True, exist_ok=True)
                with open(reward_curve_path, "w") as f:
                    json.dump(reward_curve, f, indent=2)

            print(f"   [step {step:>4}] completion_valid_rate: {valid_rate:.1%}")

    grpo_config = GRPOConfig(
        output_dir="checkpoints/faithfulreward",
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_generations=4,
        temperature=0.7,
        generation_kwargs={"max_new_tokens": 256},
        logging_steps=10,
        save_steps=50,
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

    print(f"\n🎯 Starting GRPO for {steps} steps...")
    print("   Watch completion_valid_rate — target >50% before reward improves!\n")
    trainer.train()

    save_path = "checkpoints/faithfulreward/final"
    trainer.save_model(save_path)
    print(f"\n✅ Training complete. Model saved to {save_path}")
    print(f"   Reward curve saved to {reward_curve_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaithfulReward GRPO Trainer v2")
    parser.add_argument("--data", default="data/divergence_log.jsonl")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    train(data_path=args.data, steps=args.steps, model_name=args.model)
