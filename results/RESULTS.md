# FaithfulReward — Evaluation Results

Evaluation run: 2026-07-24  
Test set: 60 held-out records (20% stratified split from 300-record divergence log)

---

## Summary: Baseline vs Trained Model

| Metric | Baseline (no RL) | Trained (50 steps) | Delta |
|---|---|---|---|
| Mean Reward | +0.0201 | −0.3000 | −0.32 |
| Faithful % | 45.0% | 0.0% | −45.0pp |
| Flagged Rate % | 46.7% | N/A | — |
| Divergence Rate % | 11.7% | N/A | — |
| n_samples | 60 | 60 | — |

> **Why trained < baseline:** All model completions were clipped at `max_new_tokens=100` without terminating. The model never emitted a valid `SCORES:` line, so every completion received the malformed-output penalty (−0.30). The baseline scores real auditor scores from the dataset; the trained model is scored on its generated text. At 50 steps the model has not yet learned to produce structured output within the token budget. See Key Findings.

---

## Domain Breakdown (Mean Reward)

Baseline only — the trained checkpoint eval did not produce per-domain scores (all completions malformed).

| Domain | Baseline Mean Reward |
|---|---|
| Math | +0.0321 |
| Ethics | +0.2364 |
| Medical | −0.2208 |

Medical is the hardest domain: causal claims and reference integrity failures pull reward negative. Ethics scores highest because many steps involve clearly-positioned philosophical arguments with high logical validity.

---

## Weakest Dimension

**Most common weak dimension (baseline): `reference_integrity`**

`reference_integrity` (threshold: 0.80) is the strictest threshold of the three and the most frequently failing. A step passes `reference_integrity` only if it cites facts already established in prior context. Reasoning steps that introduce new facts or make causal leaps without grounding — common in medical and ethics domains — fail this check.

This aligns with the domain breakdown: medical has the lowest mean reward and the most reference integrity failures (causal inference claims, generalization from trial populations).

---

## Key Findings

- **Baseline mean reward (+0.02) reflects a near-random mix of faithful/unfaithful steps.** The synthetic dataset is ~45% faithful (by human verdict), so the slight positive baseline is consistent with modest auditor–human agreement.

- **The trained model (50 steps) fails to generate valid structured output.** `completions/clipped_ratio: 1.0` in training logs confirms all completions hit the 100-token ceiling without a `SCORES:` line. The model needed more context to format correctly — increase `max_new_tokens` to 256+ and train ≥500 steps for real signal.

- **Reward variance (std ~0.15–0.26) during training suggests the model is exploring but not converging.** `frac_reward_zero_std: 0.65–0.70` means 65–70% of prompts had zero reward variance across 4 generations — the model was generating near-identical (all-malformed) completions, so GRPO had no gradient signal to exploit.

- **`reference_integrity` is the primary failure mode across all domains.** Training signal should focus on teaching the model to only reference grounded facts. Future work: add explicit reference-grounding examples to the prompt template and increase the `reference_integrity` weight in the reward function.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Algorithm | GRPO (TRL 1.9.0) |
| Training steps | 50 |
| Batch size | 1 (per device) |
| Gradient accumulation | 8 steps (effective batch = 8) |
| Learning rate | 1e-5 (cosine decay) |
| num_generations | 4 |
| max_new_tokens | 100 |
| temperature | 0.7 |
| LoRA rank | r=16, alpha=32 |
| LoRA targets | q_proj, v_proj |
| LoRA dropout | 0.05 |
| Quantization | 4-bit NF4 (bitsandbytes 0.49.2) |
| double_quant | True |
| compute_dtype | float16 |
| Hardware | NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) |
| Training time | ~13 min (50 steps) |
| Checkpoint | `checkpoints/faithfulreward/final` |

---

## What Would Improve Results

1. **More steps (500–2000):** 50 steps is proof-of-concept. Real GRPO convergence on format learning requires hundreds of steps with diverse prompts.
2. **Longer generation budget:** `max_new_tokens=256` or more so completions can reach the `SCORES:` line before truncation.
3. **Format warm-up:** SFT on a small set of correct-format examples before GRPO to bootstrap structured output.
4. **Medical-domain oversampling:** Medical has the most negative baseline reward — targeted examples would help the reward signal land.
