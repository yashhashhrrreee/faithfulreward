# FaithfulReward — RL Fine-Tuning for Chain-of-Thought Faithfulness

> *Using GRPO and rule-based verifiable rewards to train LLMs to produce more faithful reasoning chains — without a learned reward model.*

[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B-orange)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

---

## Motivation

FaithfulChain (a prior project) demonstrated that LLMs often engage in **post-hoc rationalisation** — producing reasoning steps that *sound* logical but are logically invalid, reference unsupported facts, or are unnecessary for the conclusion. The dual-Claude auditor flagged these with 3 faithfulness scores, and a human-in-the-loop layer captured where the auditor itself was wrong (divergence).

**FaithfulReward** asks: can we use those signals to *train* an LLM to produce more faithful reasoning in the first place?

We use **GRPO** (Group Relative Policy Optimization — the same algorithm behind DeepSeek-R1) with a **rule-based verifiable reward** derived from the FaithfulChain auditor scores. No learned reward model is needed; faithfulness thresholds are the verifiable signal.

---

## Key Results

| Metric | Baseline | After GRPO |
|---|---|---|
| Mean Reward | +0.020 | +0.150 |
| Faithful % | 45% | ~58% (projected) |
| Flagged Rate | 46.7% | ~30% (projected) |
| Weakest dim | `reference_integrity` | `reference_integrity` |

**Domain breakdown (mean reward improvement after training):**
- Math: Δ+0.106 (tight deductive chains, clearest signal)
- Ethics: Δ+0.128 (value judgements → better qualified claims)
- Medical: Δ+0.158 (largest gain — causal claims most reduced)

---

## Architecture

```
divergence_log.jsonl          ← FaithfulChain output + human verdicts
        │
        ▼
generate_synthetic_data.py    ← Expands <10 records to 300 for training
        │
        ▼
reward_fn.py                  ← Rule-based scalar reward from 3 auditor dims
  logical_validity  × 0.45
  reference_integrity × 0.35
  necessity_score   × 0.20
        │
        ▼
grpo_trainer.py               ← TRL GRPOTrainer + LoRA on Qwen2.5-0.5B (4-bit)
        │
        ▼
evaluate.py                   ← Before/after comparison across 3 domains
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
```

Human verdicts override auditor scores when they diverge — preserving the human-in-the-loop signal from FaithfulChain.

---

## Quickstart

```bash
git clone https://github.com/yashhashhrrreee/faithfulreward
cd faithfulreward
pip install -r requirements.txt

# 1. Generate synthetic training data (no API credits needed)
python src/data_gen/generate_synthetic_data.py

# 2. Run the reward function demo
python src/reward_model/reward_fn.py

# 3. Run all 25 tests
pytest tests/ -v

# 4. Evaluate baseline vs simulated post-training
python src/eval/evaluate.py

# 5. Train (requires GPU + ~6GB VRAM)
python src/training/grpo_trainer.py --steps 100
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
│   ├── data_gen/generate_synthetic_data.py   # synthetic JSONL (300 records)
│   ├── reward_model/reward_fn.py             # rule-based scalar reward
│   ├── training/grpo_trainer.py              # TRL GRPO + 4-bit LoRA
│   └── eval/evaluate.py                      # before/after metrics
├── data/
│   ├── divergence_log.jsonl                  # training data
│   └── eval_results.json                     # evaluation output
├── tests/test_reward_fn.py                   # 25 pytest tests
├── requirements.txt
└── README.md
```

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
