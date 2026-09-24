# FaithfulReward — RL Fine-Tuning for Chain-of-Thought Faithfulness

> _Using GRPO and rule-based verifiable rewards to train LLMs to produce more faithful reasoning chains — without a learned reward model._

[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

---

## Version History

| Version | Status      | Key Change                                                                          |
| ------- | ----------- | ----------------------------------------------------------------------------------- |
| v1.0    | complete    | Proof of concept — pipeline works, model didn't converge (truncation at 100 tokens) |
| v2.0    | complete    | SFT warmup + format reward + 256 tokens — valid_rate 0%→37%, peak reward +0.173     |
| v3.0    | in progress | 1000 steps + curriculum learning + 100-example warmup + 512 tokens                  |
| v4.0    | planned     | Real data integration — 1000+ FaithfulChain traces with human annotations           |

---

## Motivation

FaithfulChain (a prior project) demonstrated that LLMs often engage in **post-hoc rationalisation** — producing reasoning steps that _sound_ logical but are logically invalid, reference unsupported facts, or are unnecessary for the conclusion. The dual-Claude auditor flagged these with 3 faithfulness scores, and a human-in-the-loop layer captured where the auditor itself was wrong (divergence).

**FaithfulReward** asks: can we use those signals to _train_ an LLM to produce more faithful reasoning in the first place?

We use **GRPO** (Group Relative Policy Optimization — the same algorithm behind DeepSeek-R1) with a **rule-based verifiable reward** derived from the FaithfulChain auditor scores. No learned reward model is needed; faithfulness thresholds are the verifiable signal.

---

## Key Results

### V1 vs V2 Side-by-Side

| Metric                        | V1 Baseline | V1 Trained | V2 Baseline         | V2 Trained             |
| ----------------------------- | ----------- | ---------- | ------------------- | ---------------------- |
| Mean Reward                   | +0.020      | −0.300     | +0.186              | −0.300 (eval temp=0.1) |
| Faithful %                    | 45.0%       | 0.0%       | 56.7%               | 0.0% (eval temp=0.1)   |
| Flagged Rate %                | 46.7%       | N/A        | 45.0%               | N/A                    |
| Completion Valid % (training) | 0%          | 0%         | N/A                 | **15–37%**             |
| Peak Training Reward          | —           | −0.300     | —                   | **+0.173** (step 230)  |
| Training Steps                | 50          | —          | 300 + 50 SFT warmup | —                      |
| max_new_tokens                | 100         | —          | 256                 | —                      |

**V1 failure explanation:** All completions were clipped at 100 tokens before emitting a `SCORES:` line. GRPO received only −0.30 (malformed penalty) on every step — no gradient signal.

**V2 primary fix:** `max_new_tokens=256` gives completions enough budget to reach `SCORES:`. SFT warmup bootstraps the format so GRPO starts from a non-zero `completion_valid_rate`.

**Domain breakdown (mean reward, baseline):**

| Domain  | V1 Baseline | V2 Baseline |
| ------- | ----------- | ----------- |
| Math    | +0.032      | +0.339      |
| Ethics  | +0.236      | −0.116      |
| Medical | −0.221      | +0.237      |

_V2 baseline differs from V1 due to changed domain distribution (medical oversampled to 40%) and different random test split._

---

## Architecture

```
divergence_log.jsonl          ← FaithfulChain output + human verdicts
        │
        ▼
generate_synthetic_data.py    ← 500 records; medical 40%, math/ethics 30% each; diversity_check()
        │
        ▼
reward_fn.py                  ← Rule-based scalar reward from 3 auditor dims
  logical_validity  × 0.45       + format_reward_bonus (+0.1) for SCORES:+VERDICT:
  reference_integrity × 0.35
  necessity_score   × 0.20
        │
        ▼
grpo_trainer.py               ← SFT warmup → curriculum GRPO (1000 steps)
  format_warmup()                100 hand-written examples (math=40, ethics=30, medical=30)
  _get_curriculum_phase()        phase 1=math, phase 2=math+ethics, phase 3=all
  EarlyStoppingCallback          stops if valid_rate > 95% for 3 intervals
  completion_valid_rate          logged per step to data/reward_curve_v3.json
        │
        ▼
evaluate.py                   ← Before/after comparison + --temperature flag + eval_samples_v3.json
```

### Why GRPO over PPO?

GRPO (Shao et al., 2024) doesn't require a separate value/critic network. It estimates advantage by comparing rewards _within a group_ of generated completions for the same prompt. This makes it:

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

| Issue                       | V1              | V2                      | V3                              |
| --------------------------- | --------------- | ----------------------- | ------------------------------- |
| Token budget                | 100 (truncated) | 256 (clips at limit)    | 512                             |
| Format bootstrap            | None            | SFT warmup, 20 examples | SFT warmup, 100 examples        |
| Training steps              | 50              | 300                     | 1000 + early stopping           |
| Format reward               | None            | +0.1 bonus              | +0.1 bonus                      |
| Domain balance              | Equal 33/33/33% | Medical 40%             | Curriculum: math→ethics→medical |
| Training data               | 300 records     | 300 records             | 500 records                     |
| completion_valid_rate (eval) | 0% (temp=0.1)  | 0% (temp=0.1), 24% (temp=0.7) | target >50% (temp=0.1)   |
| completion_valid_rate (train)| 0%             | 15–37%                  | target >90%                     |

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

# 3. Run all 46 tests (39 original/v2 tests + 7 new v3 tests)
pytest tests/ -v

# 4. Evaluate baseline vs simulated post-training
python src/eval/evaluate.py

# 5. Quick experiment: eval existing v2 checkpoint at temp=0.7
python src/eval/evaluate.py --model checkpoints/faithfulreward/final --temperature 0.7

# 6. Train v3: 100-example SFT warmup + curriculum GRPO 1000 steps (~8-12h on RTX 3050)
python src/training/grpo_trainer.py --steps 1000

# 7. Evaluate fine-tuned model at both temperatures
python src/eval/evaluate.py --model checkpoints/faithfulreward/final --temperature 0.1
python src/eval/evaluate.py --model checkpoints/faithfulreward/final --temperature 0.7
```

---

## Hardware Requirements

| Component | Minimum        | Recommended |
| --------- | -------------- | ----------- |
| GPU VRAM  | 4GB (RTX 3050) | 8GB+        |
| RAM       | 16GB           | 32GB        |
| Storage   | 5GB            | 10GB        |

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
│   ├── divergence_log.jsonl                  # training data (500 records)
│   ├── eval_results.json                     # evaluation output
│   ├── eval_samples_v3.json                  # per-sample completions from eval
│   ├── reward_curve.json                     # v2 per-step reward curve
│   └── reward_curve_v3.json                  # v3 per-step reward + clipped_ratio
├── results/
│   ├── RESULTS.md                            # v1 evaluation report
│   ├── RESULTS_v2.md                         # v2 evaluation report
│   └── RESULTS_v3.md                         # v3 evaluation report (post-training)
├── tests/
│   ├── test_reward_fn.py                     # 39 pytest tests (reward fn + v2 coverage)
│   └── test_v3.py                            # 7 new v3 tests (curriculum, early stopping, etc.)
├── requirements.txt
└── README.md
```

---

## Research Log

| Date       | Version | Report                                  | Key Finding                                                                        |
| ---------- | ------- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| 2026-07-24 | v1.0    | [RESULTS.md](results/RESULTS.md)        | Pipeline end-to-end; convergence failed due to 100-token truncation                 |
| 2026-07-24 | v2.0    | [RESULTS_v2.md](results/RESULTS_v2.md)  | SFT warmup + 256 tokens; training valid_rate 15–37%; eval at temp=0.1 still 0%     |
| 2026-07-27 | v3.0    | [RESULTS_v3.md](results/RESULTS_v3.md)  | Quick experiment: temp=0.7 eval gives 24% valid rate — confirms temperature gap    |

---

## Connection to FaithfulChain

FaithfulReward is a direct extension of [FaithfulChain](https://github.com/yashhashhrrreee/faithfulchain):

|         | FaithfulChain                      | FaithfulReward              |
| ------- | ---------------------------------- | --------------------------- |
| Goal    | Detect post-hoc rationalisation    | Reduce it via RL training   |
| Method  | Dual-Claude auditor + human review | GRPO with verifiable reward |
| Output  | Divergence dataset (JSONL)         | Fine-tuned faithful LLM     |
| RL used | No                                 | Yes (GRPO)                  |

---

## References

- Shao et al. (2024). _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models._ (GRPO algorithm)
- Mu et al. (2024). _Rule-Based Rewards for Language Model Safety._ NeurIPS 2024.
- Ouyang et al. (2022). _Training language models to follow instructions with human feedback._ (RLHF)
- Hu et al. (2021). _LoRA: Low-Rank Adaptation of Large Language Models._

---
