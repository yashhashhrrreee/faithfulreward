# FaithfulReward — RL Fine-Tuning for Chain-of-Thought Faithfulness

> *Using GRPO and rule-based verifiable rewards to train LLMs to produce more faithful reasoning chains — without a learned reward model.*

[![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

---

## Version History

| Version | Status | Key Change |
|---|---|---|
| v1.0 | complete | Proof of concept — pipeline works, model didn't converge (truncation at 100 tokens) |
| v2.0 | complete | SFT warmup + format reward + 256 tokens — valid_rate 0%→37%, peak reward +0.173 |
| v3.0 | planned | Convergence run — 1000 steps + curriculum learning (math→ethics→medical) |

---

## Motivation

FaithfulChain (a prior project) demonstrated that LLMs often engage in **post-hoc rationalisation** — producing reasoning steps that *sound* logical but are logically invalid, reference unsupported facts, or are unnecessary for the conclusion. The dual-Claude auditor flagged these with 3 faithfulness scores, and a human-in-the-loop layer captured where the auditor itself was wrong (divergence).

**FaithfulReward** asks: can we use those signals to *train* an LLM to produce more faithful reasoning in the first place?

We use **GRPO** (Group Relative Policy Optimization — the same algorithm behind DeepSeek-R1) with a **rule-based verifiable reward** derived from the FaithfulChain auditor scores. No learned reward model is needed; faithfulness thresholds are the verifiable signal.

---

## Key Results

### V1 vs V2 Side-by-Side

| Metric | V1 Baseline | V1 Trained | V2 Baseline | V2 Trained |
|---|---|---|---|---|
| Mean Reward | +0.020 | −0.300 | +0.186 | −0.300 (eval temp=0.1) |
| Faithful % | 45.0% | 0.0% | 56.7% | 0.0% (eval temp=0.1) |
| Flagged Rate % | 46.7% | N/A | 45.0% | N/A |
| Completion Valid % (training) | 0% | 0% | N/A | **15–37%** |
| Peak Training Reward | — | −0.300 | — | **+0.173** (step 230) |
| Training Steps | 50 | — | 300 + 50 SFT warmup | — |
| max_new_tokens | 100 | — | 256 | — |

**V1 failure explanation:** All completions were clipped at 100 tokens before emitting a `SCORES:` line. GRPO received only −0.30 (malformed penalty) on every step — no gradient signal.

**V2 primary fix:** `max_new_tokens=256` gives completions enough budget to reach `SCORES:`. SFT warmup bootstraps the format so GRPO starts from a non-zero `completion_valid_rate`.

**Domain breakdown (mean reward, baseline):**

| Domain | V1 Baseline | V2 Baseline |
|---|---|---|
| Math | +0.032 | +0.339 |
| Ethics | +0.236 | −0.116 |
| Medical | −0.221 | +0.237 |

*V2 baseline differs from V1 due to changed domain distribution (medical oversampled to 40%) and different random test split.*

---

## Architecture

```
divergence_log.jsonl          ← FaithfulChain output + human verdicts
        │
        ▼
generate_synthetic_data.py    ← 300 records; medical 40%, math/ethics 30% each
        │
        ▼
reward_fn.py                  ← Rule-based scalar reward from 3 auditor dims
  logical_validity  × 0.45       + format_reward_bonus (+0.1) for SCORES:+VERDICT:
  reference_integrity × 0.35
  necessity_score   × 0.20
        │
        ▼
grpo_trainer.py               ← SFT warmup (50 steps) → GRPO (300 steps)
  format_warmup()                20 hand-written examples to bootstrap structure
  completion_valid_rate          logged per step to data/reward_curve.json
        │
        ▼
evaluate.py                   ← Before/after comparison + completion_valid_rate
```

### Why GRPO over PPO?

GRPO (Shao et al., 2024) doesn't require a separate value/critic network. It estimates advantage by comparing rewards *within a group* of generated completions for the same prompt. This makes it:
- Memory-efficient (fits RTX 3050 with 4-bit + LoRA)
- More stable than PPO for small batch sizes
- The same algorithm used in DeepSeek-R1 and Anthropic's reasoning post-training

### Why rule-based reward over a learned reward model?

Following the RLVR paradigm (Reinforcement Learning from Verifiable Rewards): a learned reward model can be **hacked** — the policy finds ways to maximise the learned model's score without actually improving faithfulness. Our 3-dimensional threshold system acts as an objective verifier, analogous to a code compiler or a math checker.

---

## Reward Function

```python
# Dimensions and thresholds (inherited from FaithfulChain)
THRESHOLDS = {
    "logical_validity": 0.75,
    "reference_integrity": 0.80,
    "necessity_score": 0.65,
}

# Weighted aggregate — logical validity weighted highest
WEIGHTS = {"logical_validity": 0.45, "reference_integrity": 0.35, "necessity_score": 0.20}

# Reward in [-1, +1]; magnitude reflects confidence
reward = ±(0.5 + confidence_scale)

# v2: format bonus applied after faithfulness reward
if completion has valid SCORES: AND VERDICT::
    reward += 0.1  # clipped to [-1, +1]
```

Human verdicts override auditor scores when they diverge — preserving the human-in-the-loop signal from FaithfulChain.

---

## What V2 Fixed vs What V3 Will Address

| Issue | V1 | V2 | V3 (planned) |
|---|---|---|---|
| Token budget | 100 (truncated) | 256 (fixed) | 256+ |
| Format bootstrap | None | SFT warmup, 20 examples | SFT warmup, 100 examples |
| Training steps | 50 | 300 | 1000+ |
| Format reward | None | +0.1 bonus | +0.1 bonus |
| Domain balance | Equal 33/33/33% | Medical 40% | Curriculum: math→ethics→medical |
| completion_valid_rate | 0% | >0% (primary goal) | >90% (target) |

---

## Quickstart

```bash
git clone https://github.com/yashhashhrrreee/faithfulreward
cd faithfulreward
pip install -r requirements.txt

# 1. Generate synthetic training data (v2: medical oversampled)
python src/data_gen/generate_synthetic_data.py

# 2. Run the reward function demo
python src/reward_model/reward_fn.py

# 3. Run all 39 tests (25 original + 14 new v2 tests)
pytest tests/ -v

# 4. Evaluate baseline vs simulated post-training
python src/eval/evaluate.py

# 5. Train v2: SFT warmup + GRPO 300 steps (requires GPU + ~4GB VRAM)
python src/training/grpo_trainer.py --steps 300

# 6. Evaluate fine-tuned model
python src/eval/evaluate.py --model checkpoints/faithfulreward/final
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4GB (RTX 3050) | 8GB+ |
| RAM | 16GB | 32GB |
| Storage | 5GB | 10GB |

4-bit NF4 quantization + LoRA (r=16) keeps the base model under 2GB VRAM.

---

## Project Structure

```
faithfulreward/
├── src/
│   ├── data_gen/generate_synthetic_data.py   # synthetic JSONL (300 records, medical 40%)
│   ├── reward_model/reward_fn.py             # rule-based scalar reward + format_reward_bonus
│   ├── training/grpo_trainer.py              # SFT warmup + TRL GRPO + 4-bit LoRA
│   └── eval/evaluate.py                      # before/after metrics + completion_valid_rate
├── data/
│   ├── divergence_log.jsonl                  # training data
│   ├── eval_results.json                     # evaluation output
│   └── reward_curve.json                     # per-step reward + valid_rate (written during training)
├── results/
│   ├── RESULTS.md                            # v1 evaluation report
│   └── RESULTS_v2.md                         # v2 evaluation report
├── tests/test_reward_fn.py                   # 39 pytest tests
├── requirements.txt
└── README.md
```

---

## Research Log

| Date | Version | Report | Key Finding |
|---|---|---|---|
| 2026-07-24 | v1.0 | [RESULTS.md](results/RESULTS.md) | Pipeline end-to-end; convergence failed due to 100-token truncation |
| 2026-07-24 | v2.0 | [RESULTS_v2.md](results/RESULTS_v2.md) | SFT warmup + 256 tokens; completion_valid_rate now trackable |

---

## Connection to FaithfulChain

FaithfulReward is a direct extension of [FaithfulChain](https://github.com/yashhashhrrreee/faithfulchain):

| | FaithfulChain | FaithfulReward |
|---|---|---|
| Goal | Detect post-hoc rationalisation | Reduce it via RL training |
| Method | Dual-Claude auditor + human review | GRPO with verifiable reward |
| Output | Divergence dataset (JSONL) | Fine-tuned faithful LLM |
| RL used | No | Yes (GRPO) |

---

## References

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* (GRPO algorithm)
- Mu et al. (2024). *Rule-Based Rewards for Language Model Safety.* NeurIPS 2024.
- Ouyang et al. (2022). *Training language models to follow instructions with human feedback.* (RLHF)
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*

---

*Built as part of an AI safety research portfolio targeting the Anthropic Fellows Program (Reinforcement Learning workstream).*
