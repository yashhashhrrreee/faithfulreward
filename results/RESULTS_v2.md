# FaithfulReward v2 — Evaluation Results

Evaluation run: 2026-07-24  
Training: 50 SFT warmup steps + 300 GRPO steps (~3.5 hours on RTX 3050)  
Test set: 60 held-out records (20% stratified split from 300-record divergence log)

---

## 1. Overview

FaithfulReward is an RL fine-tuning pipeline that trains a small LLM (Qwen2.5-0.5B-Instruct) to produce faithful chain-of-thought reasoning, using GRPO and a rule-based verifiable reward derived from FaithfulChain auditor scores. v1 proved the pipeline end-to-end but failed to converge: all completions were truncated at 100 tokens before the model could emit a `SCORES:` line, so GRPO received only the malformed-output penalty (−0.30) on every step with `completion_valid_rate=0%` throughout. v2 addressed the four root causes: generation budget (100→256 tokens), training duration (50→300 steps), format bootstrapping (50-step SFT warmup on 20 hand-written examples), and reward signal density (format bonus +0.1). The primary v2 goal was to get `completion_valid_rate > 0%` so GRPO has a format signal to exploit. That goal was achieved: the model reached 15–37% valid completions during training, compared to 0% in v1. However, deterministic evaluation (temperature=0.1) still shows 0% valid rate, revealing a second convergence bottleneck: format learning has not yet generalised from stochastic to greedy decoding.

---

## 2. V1 vs V2 Comparison

| Metric | V1 Baseline | V1 Trained (50 steps) | V2 Baseline | V2 Trained (300 steps) |
|---|---|---|---|---|
| Mean Reward | +0.020 | −0.300 | +0.186 | −0.300 |
| Faithful % | 45.0% | 0.0% | 56.7% | 0.0% |
| Flagged Rate % | 46.7% | N/A | 45.0% | N/A |
| Completion Valid % (eval, temp=0.1) | 0% | 0% | N/A | **0%** |
| Completion Valid % (training, temp=0.7) | — | 0% | — | **15–37%** |
| Peak Training Reward | — | −0.30 | — | **+0.173** (step 230) |
| Final Training Reward | — | −0.30 | — | **+0.023** (step 300) |
| Training Steps | 50 | — | 300 + 50 SFT warmup | — |
| max_new_tokens | 100 | — | 256 | — |
| SFT Warmup | No | — | Yes (50 steps, 20 examples) | — |
| Domain Distribution | 33/33/33% | — | math=30%, ethics=30%, medical=40% | — |
| Training Time | ~13 min | — | ~3.5 hours | — |

> **Key distinction between v1 and v2:** The eval metric (mean reward, faithful %) looks identical at −0.30 and 0.0%. But the training dynamics are fundamentally different. v1 had `completion_valid_rate=0%` throughout all 50 steps — GRPO had no signal whatsoever. v2 had 15–37% valid completions during training, positive rewards at steps 230, 270, 290, and 300, and `frac_reward_zero_std` dropped to 0 at the best steps — GRPO had genuine signal to exploit. The eval metric failing is a second bug: deterministic decoding at temp=0.1 doesn't reproduce the stochastic format learning acquired at temp=0.7.

---

## 3. What Changed and Why

### Fix 1: max_new_tokens 100 → 256 + 300 GRPO steps

**V1 problem:** The prompt alone is ~120 tokens. With `max_new_tokens=100`, every completion was truncated before the `SCORES:` line. `completion_valid_rate=0%`, no GRPO gradient.

**V2 change:** `max_new_tokens=256`, 300 steps.

**Effect:** Completions now reach `SCORES:` stochastically — valid rate 15–37% in training. However, `completions/clipped_ratio=1.0` throughout, meaning the model still fills all 256 tokens without natural termination. Even 256 tokens is insufficient for reliable termination. The budget fix was necessary but not sufficient.

---

### Fix 2: SFT format warmup (50 steps, 20 examples)

**V1 problem:** GRPO started from zero format knowledge. The base model had no prior for 3-line structured output.

**V2 change:** 50 SFT steps on 20 hand-written examples (math/ethics/medical, faithful and unfaithful), run before GRPO via TRL `SFTTrainer`. Loss dropped from 2.507 → 2.272 in 50 steps, token accuracy 56.5% → 58.4%.

**Effect:** The warmed-up model immediately produced 15% valid completions at GRPO step 10, vs 0% throughout all of v1. The warmup demonstrably bootstrapped format learning. Without it, v2 would likely have shown 0% valid rate for the first 100+ GRPO steps.

---

### Fix 3: Format reward bonus (+0.1)

**V1 problem:** Malformed output always returned −0.30 (flat penalty). No intermediate reward for partial format correctness.

**V2 change:** `format_reward_bonus()` in `reward_fn.py` adds +0.1 when completion has both valid `SCORES:` and `VERDICT:` lines.

**Effect:** Steeper gradient at the format boundary. Steps with higher valid_rate (80: 26%, 120: 29%, 170: 33%, 270: 37%) generally correlate with higher mean rewards. The format bonus provides a distinguishable signal between "correct format, wrong scores" and "no format at all."

---

### Fix 4: Medical domain oversampling (40%)

**V1 problem:** Medical was the hardest domain (−0.221 baseline reward in v1) with only 33% of training data.

**V2 change:** Medical oversampled to 40% via weighted `random.choices`. Dataset: math=59 (20%), ethics=104 (35%), medical=137 (46%).

**Effect:** Medical baseline reward improved to +0.237 in v2 (vs −0.221 in v1). Note: this is partly due to different test split, not purely training. Training reward impact is hard to isolate but the domain is better represented.

---

## 4. Domain Breakdown

| Domain | V1 Baseline | V2 Baseline | V2 Trained (eval) |
|---|---|---|---|
| Math | +0.032 | +0.339 | −0.300 |
| Ethics | +0.236 | −0.116 | −0.300 |
| Medical | −0.221 | +0.237 | −0.300 |

All trained domains return −0.30 in deterministic eval (all completions malformed at temp=0.1).

**Math:** V2 baseline +0.339 (up from +0.032) due to test split drawing more high-scoring math examples with tight deductive structure.

**Ethics:** V2 baseline −0.116 (down from +0.236) — more ethics examples in v2 test split include value-judgement steps with low `reference_integrity`. Ethics is the most volatile domain.

**Medical:** V2 baseline +0.237 (up from −0.221) — new test split hit more statistics-based medical steps (RCT design, p-values, confidence intervals) which score well. Medical oversampling in training may also help slightly.

---

## 5. Key Findings

- **V2 fixed the primary v1 failure.** `completion_valid_rate` rose from 0% throughout v1 to 15–37% during v2 GRPO training. GRPO finally had a non-zero gradient signal to exploit.

- **Training reward crossed positive.** Steps 230 (+0.173), 270 (+0.152), 290 (+0.016), 300 (+0.023) all show positive mean rewards — the first positive training rewards in the project's history. v1 never exceeded −0.30.

- **`frac_reward_zero_std` dropped to 0 at best steps.** At step 270 (best valid_rate=37%), `frac_reward_zero_std=0` meaning every prompt had reward variance across its 4 generations. GRPO was fully exploiting the signal.

- **Eval at temperature=0.1 still shows 0%.** The trained model produces valid completions stochastically (temp=0.7) but not deterministically (temp=0.1). Format learning acquired via high-temperature sampling hasn't generalised to greedy decoding. This is the v3 target.

- **`completions/clipped_ratio=1.0` throughout all 300 steps.** Every completion hit the 256-token ceiling without natural termination. The model generates long verbose preambles before reaching `SCORES:`. Need to either increase budget further (384+) or teach conciseness via SFT.

- **Reward variance (std ~0.48–0.64) is healthy.** GRPO is not collapsed to a degenerate policy. `reward_std` at final step 300 is 0.606, indicating the model is generating diverse outputs with varying reward.

- **300 steps is still not enough for convergence.** Reward oscillates in the range [−0.22, +0.17] rather than trending upward monotonically. Consistent positive reward would require either more steps or higher valid_rate floor.

---

## 6. Failure Analysis

**What still didn't work and why:**

1. **Deterministic eval (temp=0.1) shows 0% valid rate.** The 15–37% valid rate during training was stochastic — at high temperature, the model occasionally produces a SCORES: line. At low temperature, it reverts to verbose fluent text without structure. The LoRA update has shifted the distribution toward format-aware outputs at high temperature but not changed the greedy mode behaviour. Fix: evaluate at temp=0.7, or train further to raise valid_rate floor so even greedy decoding produces structured output.

2. **`clipped_ratio=1.0` throughout.** Even with 256 tokens, the model never terminates naturally. `completions/mean_terminated_length=0` across all 300 steps. The model is generating up to 256 tokens of text that sometimes (30%) contains a SCORES: line somewhere in the middle. This is a formatting problem: the model should learn to produce the 3-line output first, then stop. Fix: SFT warmup with explicit EOS signalling, or increase `max_new_tokens` to 512.

3. **Reward oscillation, not monotonic increase.** Steps 230 and 270 have the highest rewards (+0.17, +0.15), but step 240 drops back to −0.05. The 300-step run shows noisy convergence. Would require 1000+ steps with stable `frac_reward_zero_std=0` to see clear trend. This is a step count problem.

4. **0.5B model capacity ceiling.** Qwen2.5-0.5B may lack sufficient capacity to reliably generate structured 3-line output concisely. A 3B model would likely achieve >50% valid rate at the same step count.

5. **20 SFT warmup examples caused partial overfitting.** Token accuracy in SFT reached ~58% — not high enough to guarantee format generalisation. 100 examples (V3 target) would give the format prior more robustness.

---

## 7. Future Scope — V3 and Beyond

### V3 — Convergence (next immediate step)

Target: `completion_valid_rate > 90%` (training), deterministic eval valid rate > 50%, mean reward > +0.30

- **1000+ GRPO steps** — v2 showed positive reward at step 230+; 1000 steps would stabilise this
- **SFT warmup on 100 examples** — better coverage, less template memorisation
- **Increase `max_new_tokens` to 512** — resolve the clipping problem; even 256 isn't enough
- **Curriculum learning:** math-only for steps 0–300 (easiest domain, clearest deductive chains), add ethics at step 300, add medical at step 600
- **Evaluate at temperature=0.7** as well as 0.1 — gives honest picture of stochastic capability

---

### V4 — Real Data Integration

Target: train on 1000+ real reasoning traces with human annotations

- **Replace synthetic `divergence_log.jsonl`** with real FaithfulChain output (actual Claude API runs) — synthetic steps repeat 10–15 templates; real reasoning has much higher diversity
- **Human annotation pipeline** for divergence labels — crowdworker annotation with inter-rater agreement (Cohen's κ > 0.7 target)
- **Expand domain coverage:** add science, law, and financial reasoning — current 3-domain scope is too narrow for generalization claims
- **Dataset size:** 1000 records minimum for robust GRPO, 5000+ for publication-quality results

---

### V5 — Scale Up

Target: publish benchmark results comparing base vs RL-trained faithfulness

- **Move from Qwen2.5-0.5B to Qwen2.5-3B or Phi-3-mini** — 6× parameter increase likely enables reliable structured output without SFT warmup crutch
- **Cloud GPU (Colab A100 or Lambda Labs)** — RTX 3050 is 4GB VRAM; A100 is 80GB, enabling `num_generations=16`, batch_size=4, and higher `max_new_tokens`
- **Domain-specific reward weights:** higher `reference_integrity` weight for medical, higher `logical_validity` for math
- **Multi-run ablation:** compare no-warmup vs warmup, format bonus vs no bonus, equal vs oversampled domains — isolate which v2 change contributed most

---

### V6 — Paper Submission

Target: arXiv preprint, workshop track at NeurIPS/ICLR

- **Formal evaluation against faithfulness benchmarks:** TruthfulQA, FaithDial, BEGIN — compare base model vs RL-trained on each
- **Ablation study:** threshold values, dimension weights, OR-logic vs AND-logic flagging, format bonus magnitude
- **Comparison: rule-based reward vs learned reward model** — train a learned reward model on the same data, compare convergence speed and reward hacking rate
- **Human evaluation:** 100-sample preference study comparing base vs RL-trained outputs on faithfulness
- **Reproducibility:** full training code, model weights, dataset released on HuggingFace Hub

---

## 8. Training Configuration

| Parameter | V1 | V2 |
|---|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Algorithm | GRPO (TRL 1.9.0) | SFT warmup → GRPO (TRL 1.9.0) |
| SFT warmup steps | — | 50 (20 hand-written examples) |
| GRPO training steps | 50 | 300 |
| Batch size (per device) | 1 | 1 |
| Gradient accumulation | 8 (eff. batch = 8) | 8 (eff. batch = 8) |
| Learning rate (GRPO) | 1e-5 | 1e-5 |
| Learning rate (SFT) | — | 2e-5 |
| num_generations | 4 | 4 |
| max_new_tokens | 100 | 256 |
| temperature | 0.7 | 0.7 |
| LoRA rank | r=16, alpha=32 | r=16, alpha=32 |
| LoRA targets | q_proj, v_proj | q_proj, v_proj |
| LoRA dropout | 0.05 | 0.05 |
| Quantization | 4-bit NF4 | 4-bit NF4 |
| double_quant | True | True |
| compute_dtype | float16 | float16 |
| Hardware | RTX 3050 Laptop (4 GB VRAM) | RTX 3050 Laptop (4 GB VRAM) |
| Training time | ~13 min | ~3.5 hours |
| Domain distribution | 33/33/33% | math=30%, ethics=30%, medical=40% |
| Format reward bonus | No | +0.1 for SCORES:+VERDICT: |
| SFT warmup token accuracy | — | 56.5% → 58.4% |
| Peak training reward | −0.300 | +0.173 (step 230) |
| Final training reward | −0.300 | +0.023 (step 300) |

---

## 9. Reproducibility

Complete commands to reproduce the v2 run from scratch:

```bash
# 0. Clone and setup
git clone https://github.com/yashhashhrrreee/faithfulreward
cd faithfulreward
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate synthetic training data (v2: medical 40%)
python src/data_gen/generate_synthetic_data.py

# 2. Run all 39 tests (25 original + 14 new v2 tests)
pytest tests/ -v

# 3. Run baseline evaluation
python src/eval/evaluate.py --data data/divergence_log.jsonl

# 4. Train: 50 SFT warmup + 300 GRPO steps (~3.5 hours on RTX 3050)
python src/training/grpo_trainer.py \
    --data data/divergence_log.jsonl \
    --steps 300 \
    --model Qwen/Qwen2.5-0.5B-Instruct

# 5. Evaluate fine-tuned model
python src/eval/evaluate.py \
    --data data/divergence_log.jsonl \
    --model checkpoints/faithfulreward/final

# 6. View reward curve
python -c "
import json
curve = json.load(open('data/reward_curve.json'))
for entry in curve:
    print(f\"Step {entry['step']:>4}: reward={entry['mean_reward']:+.4f}  valid_rate={entry['completion_valid_rate']:.1%}\")
"
```

**Requirements:** NVIDIA GPU with 4GB+ VRAM, CUDA 12.1+, Python 3.11+. Qwen2.5-0.5B-Instruct downloads automatically (~1GB). bitsandbytes requires a compatible CUDA driver.
