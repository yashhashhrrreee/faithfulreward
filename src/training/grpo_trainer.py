"""
grpo_trainer.py
---------------
Fine-tunes a small LLM (Qwen2.5-0.5B) using GRPO (Group Relative Policy Optimization)
with FaithfulReward's rule-based reward function as the verifiable signal.

GRPO is the same RL algorithm used in DeepSeek-R1. It's simpler than PPO
because it doesn't need a separate value/critic network — it estimates
advantage by comparing rewards within a group of generated outputs.

Hardware target: RTX 3050 (4-8GB VRAM) via 4-bit quantization + LoRA

Run:
    python src/training/grpo_trainer.py --data data/divergence_log.jsonl --steps 100

Dependencies:
    pip install transformers trl peft bitsandbytes torch accelerate datasets
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reward_model.reward_fn import compute_reward


# ── Reward function wrapper for TRL ──────────────────────────────────────────
def faithfulness_reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    TRL-compatible reward function.

    TRL's GRPOTrainer calls this with a list of model completions.
    We parse the completion for auditor scores and return scalar rewards.

    The model is trained to output reasoning steps in this format:
        STEP: <reasoning text>
        SCORES: logical_validity=0.85, reference_integrity=0.90, necessity_score=0.75
        VERDICT: faithful

    If parsing fails, return a small negative reward (encourage valid format).
    """
    rewards = []
    for completion in completions:
        try:
            reward = _parse_and_score(completion)
        except Exception:
            reward = -0.3  # mild penalty for malformed output
        rewards.append(reward)
    return rewards


def _parse_and_score(completion: str) -> float:
    """Parse model output and compute reward."""
    lines = completion.strip().split("\n")
    scores = {}

    for line in lines:
        line = line.strip()
        if line.startswith("SCORES:"):
            # Parse: SCORES: logical_validity=0.85, reference_integrity=0.90, necessity_score=0.75
            score_part = line.replace("SCORES:", "").strip()
            for item in score_part.split(","):
                item = item.strip()
                if "=" in item:
                    key, val = item.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key in ("logical_validity", "reference_integrity", "necessity_score"):
                        scores[key] = float(val)

    if len(scores) == 3:
        result = compute_reward(scores)
        return result.reward
    else:
        return -0.3  # incomplete output


# ── Dataset preparation ───────────────────────────────────────────────────────
def load_training_prompts(jsonl_path: str, max_samples: int = 200) -> list[dict]:
    """
    Convert divergence log records into training prompts for GRPO.

    Each prompt asks the model to evaluate a reasoning step and produce
    faithfulness scores + verdict in structured format.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    prompts = []
    for rec in records[:max_samples]:
        prompt = _build_prompt(rec["step_text"], rec["domain"])
        # Reference output (what a faithful response looks like)
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


# ── Main training function ────────────────────────────────────────────────────
def train(data_path: str, steps: int = 100, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Run GRPO fine-tuning. Imports are inside this function so the script
    can be imported without requiring GPU libraries.
    """
    print(f"🚀 FaithfulReward GRPO Training")
    print(f"   Model      : {model_name}")
    print(f"   Data       : {data_path}")
    print(f"   Steps      : {steps}")
    print()

    # ── Imports (heavy, GPU-required) ─────────────────────────────────────────
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install transformers trl peft bitsandbytes torch accelerate datasets")
        sys.exit(1)

    # ── 4-bit quantization config (fits RTX 3050) ────────────────────────────
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

    # ── LoRA config — lightweight adapter only ────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # attention heads only
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Load and prepare dataset ──────────────────────────────────────────────
    print("📂 Loading training prompts...")
    prompt_dicts = load_training_prompts(data_path, max_samples=200)
    dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in prompt_dicts])
    print(f"   {len(dataset)} training examples loaded")

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir="checkpoints/faithfulreward",
        max_steps=steps,
        per_device_train_batch_size=1,       # small batch for 4-8GB VRAM
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_generations=4,                   # GRPO generates 4 completions per prompt
        temperature=0.7,
        generation_kwargs={"max_new_tokens": 100},
        logging_steps=10,
        save_steps=50,
        report_to="none",                    # disable wandb for now
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=faithfulness_reward_fn,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"\n🎯 Starting GRPO training for {steps} steps...")
    print("   Watch for 'reward' increasing over steps — that's the signal!\n")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = "checkpoints/faithfulreward/final"
    trainer.save_model(save_path)
    print(f"\n✅ Training complete. Model saved to {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaithfulReward GRPO Trainer")
    parser.add_argument("--data", default="data/divergence_log.jsonl", help="Path to divergence log JSONL")
    parser.add_argument("--steps", type=int, default=100, help="Number of GRPO training steps")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    args = parser.parse_args()

    train(data_path=args.data, steps=args.steps, model_name=args.model)
