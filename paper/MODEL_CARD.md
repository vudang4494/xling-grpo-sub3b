# Model Card — LoRA Adapters for Beyond English-Only GRPO

**Vu Dang** · Independent Researcher · [`vu.dh4494@gmail.com`](mailto:vu.dh4494@gmail.com)

Paper: [arXiv:2503.16219](https://arxiv.org/abs/2503.16219) · DOI: [10.5281/zenodo.20061328](https://doi.org/10.5281/zenodo.20061328)

---

## Overview

This repository contains **10 LoRA adapter checkpoints** trained with GRPO on
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` across 4 experimental arms
(A1–A4) and 3 random seeds, plus a base model evaluation run.
The paper studies **training language** and **auxiliary rewards** as implicit
regularizers for sub-3B math reasoning.

All adapters use **LoRA r=16, α=32, dropout=0.05**, targeting
`q_proj`, `k_proj`, `v_proj`, `o_proj`. Training via TRL GRPOTrainer with
in-process vLLM 0.7.2 rollouts on 1× NVIDIA A100-SXM4-80GB.

---

## Released Adapters

| Path | Arm | Training Data | Rewards | Seed |
|---|---|---|---|---|
| `results/grpo/reproduce_openrs_rs2_42/checkpoint-50/` | A1 (EN baseline) | `knoveleng/open-rs` 7K EN | R1+R2 | 42 |
| `results/grpo/reproduce_openrs_rs2_123/checkpoint-50/` | A1 | `knoveleng/open-rs` 7K EN | R1+R2 | 123 |
| `results/grpo/reproduce_openrs_rs2_7/checkpoint-50/` | A1 | `knoveleng/open-rs` 7K EN | R1+R2 | 7 |
| `results/grpo/a2_vi_42/checkpoint-50/` | A2 (VI training) | `5CD-AI/VI-MetaMathQA` 7K VI | R1+R2 | 42 |
| `results/grpo/a2_vi_123/checkpoint-50/` | A2 | `5CD-AI/VI-MetaMathQA` 7K VI | R1+R2 | 123 |
| `results/grpo/a2_vi_7/checkpoint-50/` | A2 | `5CD-AI/VI-MetaMathQA` 7K VI | R1+R2 | 7 |
| `results/grpo/a3_enlang_42/checkpoint-50/` | A3 (EN + R5) | `knoveleng/open-rs` 7K EN | R1+R2+R5 | 42 |
| `results/grpo/a3_enlang_123/checkpoint-50/` | A3 | `knoveleng/open-rs` 7K EN | R1+R2+R5 | 123 |
| `results/grpo/a3_enlang_7/checkpoint-50/` | A3 | `knoveleng/open-rs` 7K EN | R1+R2+R5 | 7 |
| `results/grpo/a4_const_bias_42/checkpoint-50/` | A4 (mechanism ablation) | `knoveleng/open-rs` 7K EN | R1+R2+1.0 | 42 |
| `results/grpo/a4_const_bias_123/checkpoint-50/` | A4 | `knoveleng/open-rs` 7K EN | R1+R2+1.0 | 123 |
| `results/grpo/a4_const_bias_7/checkpoint-50/` | A4 | `knoveleng/open-rs` 7K EN | R1+R2+1.0 | 7 |

Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (evaluated directly, no training).

### Reward Definitions

| Reward | Description |
|---|---|
| **R1** — correctness | Math-Verify symbolic equivalence check |
| **R2** — format | Presence of `<think>...</think>` + `<answer>...</answer>` tags |
| **R5** — language consistency | fastText `lid.176.bin` language match (fires on all prompts; EN=0 no-penalty guard disabled) |
| **R_const** — constant bias | Always 1.0; tests reward-magnitude hypothesis |

---

## Evaluation Results (3 seeds: 42, 123, 7 · mean ± σ)

| Arm | AMC-23 pass@1 | AMC-23 maj@4 | MATH-500 pass@1 | AIME-2024 pass@1 | AIME-2024 maj@8 |
|---|---|---|---|---|---|
| **Base** (untrained) | 50.0 | 70.0 | 59.4 | 26.7 | 33.3 |
| **A1** — EN baseline | 56.7 ± 11.3 | 70.0 ± 4.3 | 59.7 ± 1.7 | 14.4 ± 7.7 | 32.2 ± 6.9 |
| **A2** — VI training | 56.7 ± 5.2 | 70.0 ± 2.5 | 60.8 ± 1.0 | 20.0 ± 5.8 | 34.4 ± 1.9 |
| **A3** — EN + R5 | **57.5 ± 6.6** | 68.3 ± 3.8 | 60.7 ± 0.6 | 21.1 ± 1.9 | **37.8 ± 1.9** |
| **A4** — constant bias | 52.5 ± 0.0 | 70.0 ± 2.5 | 61.2 ± 0.9 | 24.4 ± 1.9 | 32.2 ± 5.1 |

**Key findings:**
- **A3 achieves the best AIME-2024 maj@8** (37.8 ± 1.9%, +4.5 pp over base) — the largest positive Δ on the hardest benchmark (A2 also gains +1.1 pp).
- **A3 − A4 = +5.6 pp** on AIME-2024 maj@8 (95% bootstrap CI [+1.1, +11.1]), confirming R5's contribution is **content-specific** (not pure reward-magnitude).
- **No cross-lingual transfer**: all arms land within 0.5 pp of base (a 0.7 pp band) on the MGSM 10-language benchmark.
- **Vanilla EN GRPO (A1) is unstable**: σ = 11.3 pp on AMC-23 pass@1.

All eval JSONs with full `responses[]` arrays are released in `results/eval/`.

---

## Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Example: load A3 arm, seed=42
model = PeftModel.from_pretrained(
    base,
    "results/grpo/a3_enlang_42/checkpoint-50/"
)

prompt = tok.apply_chat_template(
    [{"role": "user", "content": (
        "Solve the following math problem efficiently and clearly. "
        "The last line of your response should be of the following format: "
        "'Therefore, the final answer is: $\\boxed{ANSWER}$.'\n"
        "Think step by step.\n\n"
        "What is the sum of all positive integers less than 100 divisible by 7?"
    )}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.0)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

---

## Limitations

- 3 seeds × 4 arms on a single A100 is under-powered for statistical
  significance on low-sample benchmarks (AMC-23: n=40, AIME-2024: n=30).
- LoRA r=16 has lower expressive capacity than the full-parameter Open-RS
  setup; absolute numbers differ from Open-RS paper claims.
- A2 dataset (5CD-AI VI translation) confounds training language with
  data distribution differences from Open-RS.
- No cross-lingual transfer observed at this scale — this is a genuine
  negative finding, not a training bug.
- A4 MGSM evaluation used only seed=42; A1/A2/A3 used seeds 123 and 7
  (seed=42 not run, disclosed in table captions).

---

## License

LoRA adapter weights: Apache-2.0 (same as code license).
Manuscript and figures: CC BY 4.0.
Evaluation JSONs (with `responses[]`): CC BY 4.0.
