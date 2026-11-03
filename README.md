# ML Frontier

> Language models from scratch: tokenizers, architectures, training systems, scaling laws, post-training and evaluation.

This repository is Phase 5 of my Applied ML track. It follows Stanford **CS336** (Language Modeling from Scratch), together with current work on LLM evaluation, scaling and reliability.

It covers what it takes to build a language model rather than call one: a byte-pair tokenizer, a modern decoder architecture with mixed precision, the memory, kernel and parallelism work that makes training fit on hardware, scaling-law and compute-optimal experiments on a small model, supervised fine-tuning and DPO, and evaluation that resists contamination and judge bias.

## At a glance

| | |
|---|---|
| Track | Applied ML — Phase 5 — Frontier & Systems |
| Projects | 8, about 22 days of work |
| Tasks | 39 (31 code, 8 written; 8 advanced) |
| Verification | 15 automated tests, 24 metric targets |
| Courses | CS336 Language Modeling from Scratch (2025), LLM Evaluation & Scaling |
| Textbooks | *The Elements of Statistical Learning (2nd ed., 12th printing)* (Hastie, Tibshirani & Friedman) |

## By the end of this repository I will be able to

- Implement a byte-pair encoding tokenizer from scratch and measure its compression
- Build a modern LM architecture (RMSNorm, SwiGLU, RoPE) and train it in mixed precision
- Profile and reduce training memory and time with kernels, activation checkpointing and parallelism
- Fit scaling laws on a family of small models and reproduce the Chinchilla compute-optimal result
- Post-train a model with SFT and DPO
- Evaluate models on benchmarks with contamination checks, and measure the reliability of LLM-as-judge and red-teaming

## Projects

| Directory | Project | Days | Lectures | Tasks | Status |
|---|---|---|---|---|---|
| [`0x00-bpe_tokenizer`](./0x00-bpe_tokenizer) | [Byte-Pair Encoding Tokenizer from Scratch](#0x00-bpe_tokenizer--byte-pair-encoding-tokenizer-from-scratch) | 3 | CS336 Lec 1–2 | 3 + 1 advanced | todo |
| [`0x01-modern_lm`](./0x01-modern_lm) | [Modern LM Architecture & Mixed Precision](#0x01-modern_lm--modern-lm-architecture--mixed-precision) | 3 | CS336 Lec 3–4 | 5 + 1 advanced | todo |
| [`0x02-training_systems`](./0x02-training_systems) | [Training Systems: Memory, Kernels & Parallelism](#0x02-training_systems--training-systems-memory-kernels--parallelism) | 2 | CS336 Lec 5–6 | 3 + 1 advanced | todo |
| [`0x03-scaling_laws`](./0x03-scaling_laws) | [Scaling Laws: Train a Small LM](#0x03-scaling_laws--scaling-laws-train-a-small-lm) | 4 | CS336 Lec 7–8 | 4 + 1 advanced | todo |
| [`0x04-post_training`](./0x04-post_training) | [Post-training at LM Scale: SFT & DPO](#0x04-post_training--post-training-at-lm-scale-sft--dpo) | 3 | CS336 Lec 9–10 | 5 + 1 advanced | todo |
| [`0x05-llm_evaluation`](./0x05-llm_evaluation) | [LLM Evaluation: Benchmarks & Contamination](#0x05-llm_evaluation--llm-evaluation-benchmarks--contamination) | 3 | EVAL Evaluation fundamentals | 4 + 1 advanced | todo |
| [`0x06-llm_judge`](./0x06-llm_judge) | [LLM-as-Judge, Reliability & Red-Teaming](#0x06-llm_judge--llm-as-judge-reliability--red-teaming) | 2 | EVAL LLM-as-judge & reliability | 4 + 1 advanced | todo |
| [`0x07-chinchilla`](./0x07-chinchilla) | [Compute-Optimal Training: Chinchilla](#0x07-chinchilla--compute-optimal-training-chinchilla) | 2 | EVAL Scaling laws in depth | 3 + 1 advanced | todo |

Status moves from `todo` to `in progress` to `done` as each project is submitted and passes its checks.

## Repository layout

```
ml-frontier/
├── README.md
├── 0x00-bpe_tokenizer/
│   ├── README.md
│   ├── 0-bpe.py
│   ├── 1-tokenize_corpus.py
│   ├── merges.json
│   ├── results.json
│   ├── written/2-tokenizer_comparison.md
│   └── 100-special_tokens.py
├── 0x01-modern_lm/
│   ├── README.md
│   ├── 0-rmsnorm_swiglu.py
│   ├── 1-rope.py
│   ├── 2-gqa.py
│   ├── 3-modern_transformer.py
│   ├── results.json
│   ├── 4-benchmark.py
│   └── … 2 more
├── 0x02-training_systems/
│   ├── README.md
│   ├── 0-memory.py
│   ├── 1-memory_experiment.py
│   ├── results.json
│   ├── written/2-zero_fsdp_notes.md
│   └── 100-fused_rmsnorm.py
├── 0x03-scaling_laws/
│   ├── README.md
│   ├── 0-power_law.py
│   ├── 1-lr_schedule.py
│   ├── 2-train_lm.py
│   ├── results.json
│   ├── loss_curves.png
│   ├── written/3-scaling_report.md
│   └── … 1 more
├── 0x04-post_training/
│   ├── README.md
│   ├── 0-sft_data.py
│   ├── 1-sft.py
│   ├── results.json
│   ├── sft_samples.md
│   ├── 2-dpo.py
│   ├── preference_sample.jsonl
│   └── … 3 more
├── 0x05-llm_evaluation/
│   ├── README.md
│   ├── 0-mc_eval.py
│   ├── 1-contamination.py
│   ├── 2-evaluate.py
│   ├── results.json
│   ├── written/3-eval_report.md
│   └── 100-semantic_contamination.py
├── 0x06-llm_judge/
│   ├── README.md
│   ├── 0-agreement.py
│   ├── 1-position_bias.py
│   ├── 2-judge_experiment.py
│   ├── results.json
│   ├── written/3-capability_eval.md
│   └── 100-red_team.py
└── 0x07-chinchilla/
    ├── README.md
    ├── 0-chinchilla.py
    ├── written/2-my_runs.md
    └── 100-fit_approach3.py
```

## Conventions

- Python 3.10+. Every file starts with `#!/usr/bin/env python3`, ends with a newline, and has a docstring on every function. Nothing runs on import; demos live under `if __name__ == "__main__":`.
- Each task ships a `N-main.py` style example in its project README, so any function can be run on its own.
- Training scripts write their final metrics and configuration to `results.json` in the project directory; notebooks are committed executed, with outputs visible.
- Written answers live in `written/` as Markdown with LaTeX.
- Every project directory has its own `README.md`: what it contains, one line per file, and how to run it.

## Setup

```bash
git clone https://github.com/Olagold-hackxx/ml-frontier.git
cd ml-frontier
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib pandas jupyter torch torchvision
cd 0x00-bpe_tokenizer   # each project README shows how to run its examples
```

## Project details

### 0x00-bpe_tokenizer — Byte-Pair Encoding Tokenizer from Scratch

**3 days** · CS336 Lec 1–2

**Covers:** Course overview, tokenization (characters, bytes, words, BPE), data pipelines, PyTorch & resource accounting.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Byte-level BPE: train, encode, decode | mandatory · code | `0-bpe.py` |
| Tokenize a 100 MB corpus | mandatory · code | `1-tokenize_corpus.py`, `merges.json`, `results.json` |
| Compare with GPT-2's tokenizer | mandatory · written | `written/2-tokenizer_comparison.md` |
| Special tokens | advanced · code | `100-special_tokens.py` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `corpus_mb` ≥ 100 (Tokenize a 100 MB corpus), read from `results.json`
- `bytes_per_token` ≥ 3 (Tokenize a 100 MB corpus), read from `results.json`
- `encode_mb_per_s` ≥ 0.2 (Tokenize a 100 MB corpus), read from `results.json`

**What I will be able to do**

- Implement byte-level BPE training, encoding and decoding in pure Python
- Explain why byte-level BPE never produces an unknown token
- Explain why encoding must apply merges in rank order
- Tokenize a 100 MB corpus efficiently and report compression and throughput
- Compare your tokenizer with GPT-2's tiktoken encoding

**Key concepts:** Byte-level vocabularies: 256 base tokens, no unknown token · BPE training: iteratively merge the most frequent adjacent pair · Merge ranks and greedy encoding in merge order · Round-trip property decode(encode(s)) == s · Pre-tokenization (GPT-2 regex) and special tokens · Compression ratio (bytes per token) and throughput

<details><summary>Rules and resources</summary>

Rules:

- Pure Python (standard library only) for 0-bpe.py and 100-special_tokens.py — no tokenizers/tiktoken/sentencepiece inside them
- tiktoken allowed only in the comparison script

Resources:

- [CS336 Spring 2025 course site (assignment 1: BPE tokenizer)](https://stanford-cs336.github.io/spring2025/)
- [Sennrich, Haddow & Birch 2015 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Karpathy — minbpe (minimal byte-level BPE)](https://github.com/karpathy/minbpe)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)
- [Eldan & Li 2023 — TinyStories](https://arxiv.org/abs/2305.07759)

</details>

### 0x01-modern_lm — Modern LM Architecture & Mixed Precision

**3 days** · CS336 Lec 3–4

**Covers:** Architecture & hyper-parameter choices (pre-norm, RMSNorm, SwiGLU, RoPE, GQA/MQA), mixture of experts, mixed precision.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| RMSNorm and SwiGLU | mandatory · code | `0-rmsnorm_swiglu.py` |
| Rotary position embeddings | mandatory · code | `1-rope.py` |
| Grouped-query attention | mandatory · code | `2-gqa.py` |
| Refactor your Phase 3 transformer to a modern block | mandatory · code | `3-modern_transformer.py`, `results.json` |
| Mixed-precision speed benchmark | mandatory · code | `4-benchmark.py`, `results.json` |
| Mixture-of-Experts FFN | advanced · code | `100-moe.py`, `moe_load.png` |

**Targets it has to hit**

- 3 automated Python tests against reference implementations and closed-form answers
- `val_loss_modern` ≤ 2.5 (Refactor your Phase 3 transformer to a modern block), read from `results.json`
- `speedup_bf16` ≥ 1.2 (Mixed-precision speed benchmark), read from `results.json`
- `tokens_per_sec_fp32` > 0 (Mixed-precision speed benchmark), read from `results.json`
- `val_loss_moe` > 0 (Mixture-of-Experts FFN, advanced), read from `results.json`

**What I will be able to do**

- Implement RMSNorm, SwiGLU, RoPE and GQA in NumPy and verify them
- Prove that RoPE makes q·k depend only on the relative position
- Explain why GQA shrinks the KV-cache and by how much
- Refactor a Phase 3 transformer to a LLaMA-style block
- Benchmark fp32 vs bf16 autocast and explain the speed-up

**Key concepts:** Pre-norm vs post-norm residual blocks · RMSNorm (no mean subtraction, no bias) · SwiGLU feed-forward and the 8/3·d hidden size · Rotary position embeddings and relative-position dot products · Grouped-query attention and the KV-cache · fp32 vs fp16 vs bf16: range, precision, loss scaling

<details><summary>Rules and resources</summary>

Rules:

- NumPy only for tasks 0–3
- PyTorch for the refactor; no nn.Transformer / nn.MultiheadAttention

Resources:

- [Zhang & Sennrich 2019 — Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Shazeer 2020 — GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Su et al. 2021 — RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Ainslie et al. 2023 — GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Touvron et al. 2023 — LLaMA](https://arxiv.org/abs/2302.13971)
- [Micikevicius et al. 2017 — Mixed Precision Training](https://arxiv.org/abs/1710.03740)

</details>

### 0x02-training_systems — Training Systems: Memory, Kernels & Parallelism

**2 days** · CS336 Lec 5–6

**Covers:** GPUs and kernels (memory hierarchy, FlashAttention, Triton), parallelism (data, tensor, pipeline, ZeRO/FSDP), activation checkpointing.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Memory accounting calculator | mandatory · code | `0-memory.py` |
| Checkpointing and gradient accumulation experiment | mandatory · code | `1-memory_experiment.py`, `results.json` |
| Paper notes: ZeRO and FSDP | mandatory · written | `written/2-zero_fsdp_notes.md` |
| A fused kernel | advanced · code | `100-fused_rmsnorm.py` |

**Targets it has to hit**

- 1 automated Python test against reference implementations and closed-form answers
- `ckpt_memory_reduction` ≥ 0.3 (Checkpointing and gradient accumulation experiment), read from `results.json`
- `grad_accum_max_abs_diff` ≤ 0.0001 (Checkpointing and gradient accumulation experiment), read from `results.json`
- `fused_rmsnorm_speedup` > 1 (A fused kernel, advanced), read from `results.json`

**What I will be able to do**

- Compute the memory footprint of training a transformer from its shape
- Explain the per-GPU memory of ZeRO stages 1–3
- Measure the memory saved and time lost by activation checkpointing
- Verify that gradient accumulation reproduces large-batch gradients
- Explain why FlashAttention is faster despite doing more FLOPs

**Key concepts:** Model-state memory: parameters, gradients, optimizer states (16 bytes/param for mixed-precision Adam) · Activation memory and its dependence on sequence length and batch · Activation (gradient) checkpointing: compute ↔ memory trade-off · Gradient accumulation · ZeRO stages 1/2/3 and FSDP · Arithmetic intensity, memory-bound vs compute-bound kernels, kernel fusion

<details><summary>Rules and resources</summary>

Rules:

- NumPy / pure Python for task 0
- PyTorch on a single GPU for task 1

Resources:

- [Rajbhandari et al. 2019 — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Zhao et al. 2023 — PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [Chen et al. 2016 — Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [Korthikanti et al. 2022 — Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [Dao et al. 2022 — FlashAttention](https://arxiv.org/abs/2205.14135)
- [PyTorch docs — torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)

</details>

### 0x03-scaling_laws — Scaling Laws: Train a Small LM

**4 days** · CS336 Lec 7–8

**Covers:** Scaling laws (Kaplan, Chinchilla, IsoFLOP fits), data sources, filtering, deduplication, curriculum.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Power-law fits | mandatory · code | `0-power_law.py` |
| Warm-up + cosine LR schedule | mandatory · code | `1-lr_schedule.py` |
| Train LMs at three or more sizes on TinyStories | mandatory · code | `2-train_lm.py`, `results.json`, `loss_curves.png` |
| Scaling plot and report | mandatory · written | `written/3-scaling_report.md` |
| Mini IsoFLOP study | advanced · code | `100-isoflop.py` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `n_sizes` ≥ 3 (Train LMs at three or more sizes on TinyStories), read from `results.json`
- `params_large` ≥ 10000000 (Train LMs at three or more sizes on TinyStories), read from `results.json`
- `val_loss_large` ≤ 1.8 (Train LMs at three or more sizes on TinyStories), read from `results.json`
- `isoflop_tokens_per_param` > 0 (Mini IsoFLOP study, advanced), read from `results.json`

**What I will be able to do**

- Fit pure and saturating power laws and explain the role of the irreducible loss E
- Implement a warm-up + cosine learning-rate schedule
- Train decoder-only LMs at ≥ 3 sizes with a controlled recipe
- Plot and interpret a small scaling curve, including its uncertainty
- Explain why Kaplan and Chinchilla reached different conclusions

**Key concepts:** Power laws L(N) = a N^{−α} and saturating forms L = E + a N^{−α} · Log–log linear regression · Non-embedding parameter counts · Learning-rate warm-up and cosine decay · Data curation: deduplication and filtering · Extrapolation risk of fitted scaling curves

<details><summary>Rules and resources</summary>

Rules:

- NumPy only for tasks 0–1
- PyTorch; one consumer GPU / Colab (A100 hours optional)

Resources:

- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. 2022 — Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Eldan & Li 2023 — TinyStories](https://arxiv.org/abs/2305.07759)
- [Penedo et al. 2024 — The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [Loshchilov & Hutter 2017 — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)

</details>

### 0x04-post_training — Post-training at LM Scale: SFT & DPO

**3 days** · CS336 Lec 9–10

**Covers:** Post-training: instruction tuning / SFT, RLHF, DPO and verifiable-reward RL at LM scale.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| SFT example construction and masked loss | mandatory · code | `0-sft_data.py` |
| SFT your small LM | mandatory · code | `1-sft.py`, `results.json`, `sft_samples.md` |
| Construct preferences and run DPO | mandatory · code | `2-dpo.py`, `preference_sample.jsonl`, `results.json` |
| Win-rate evaluation | mandatory · code | `3-win_rate.py`, `results.json` |
| Post-training report | mandatory · written | `written/4-post_training_report.md` |
| RL with a verifiable reward | advanced · code | `100-grpo.py` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `sft_train_examples` ≥ 10000 (SFT your small LM), read from `results.json`
- `sft_val_loss` > 0 (SFT your small LM), read from `results.json`
- `dpo_pairs` ≥ 2000 (Construct preferences and run DPO), read from `results.json`
- `dpo_val_reward_accuracy` ≥ 0.65 (Construct preferences and run DPO), read from `results.json`
- `dpo_win_rate_vs_sft` ≥ 0.55 (Win-rate evaluation), read from `results.json`
- `grpo_constraint_success` > 0 (RL with a verifiable reward, advanced), read from `results.json`

**What I will be able to do**

- Build SFT examples with prompt-token masking and compute the masked next-token loss
- Supervised-fine-tune your own small LM on instruction data
- Construct a preference dataset for your model and run DPO on it
- Evaluate SFT vs DPO with a pairwise win rate and a standard error
- Identify failure modes: verbosity bias, reward hacking of the judge, over-optimization

**Key concepts:** Chat/instruction templates · Loss masking: train only on response tokens · SFT as maximum likelihood on demonstrations · Constructing preference pairs (rejection sampling, perturbations, AI feedback) · DPO at LM scale with a frozen reference · Pairwise win-rate evaluation with confidence intervals

<details><summary>Rules and resources</summary>

Rules:

- NumPy only for tasks 0 and 3
- PyTorch; start from your A43 largest model

Resources:

- [Ouyang et al. 2022 — InstructGPT](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. 2023 — Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Zhou et al. 2023 — LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Zheng et al. 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [CS336 Spring 2025 course site (alignment assignment)](https://stanford-cs336.github.io/spring2025/)

</details>

### 0x05-llm_evaluation — LLM Evaluation: Benchmarks & Contamination

**3 days** · EVAL Evaluation fundamentals · Reading: ESL II §7.11

**Covers:** Why evaluation is hard; MMLU, HellaSwag, BIG-bench; log-likelihood vs generative scoring; test-set contamination.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Multiple-choice accuracy with bootstrap CIs | mandatory · code | `0-mc_eval.py` |
| n-gram contamination check | mandatory · code | `1-contamination.py` |
| Evaluate your LM on 2–3 benchmarks | mandatory · code | `2-evaluate.py`, `results.json` |
| Honest evaluation report | mandatory · written | `written/3-eval_report.md` |
| Embedding-based contamination | advanced · code | `100-semantic_contamination.py` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `n_benchmarks` ≥ 2 (Evaluate your LM on 2–3 benchmarks), read from `results.json`
- `total_examples` ≥ 2000 (Evaluate your LM on 2–3 benchmarks), read from `results.json`

**What I will be able to do**

- Score multiple-choice benchmarks with log-likelihoods and length normalization
- Compute bootstrap confidence intervals and decide whether a difference is significant
- Implement an n-gram contamination check against training data
- Evaluate your own LM on 2–3 benchmarks and report honestly against chance

**Key concepts:** Multiple-choice scoring by log-likelihood (raw vs length-normalized) · Accuracy, chance level, and bootstrap confidence intervals · Prompt format sensitivity and few-shot examples · n-gram overlap contamination detection · Benchmark saturation and construct validity

<details><summary>Rules and resources</summary>

Rules:

- NumPy / standard library only for tasks 0–1
- PyTorch for evaluating your LM

Resources:

- [Hendrycks et al. 2020 — Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300)
- [Zellers et al. 2019 — HellaSwag](https://arxiv.org/abs/1905.07830)
- [Srivastava et al. 2022 — Beyond the Imitation Game (BIG-bench)](https://arxiv.org/abs/2206.04615)
- [Brown et al. 2020 — Language Models are Few-Shot Learners (13-gram contamination analysis)](https://arxiv.org/abs/2005.14165)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Miller 2024 — Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)

</details>

### 0x06-llm_judge — LLM-as-Judge, Reliability & Red-Teaming

**2 days** · EVAL LLM-as-judge & reliability

**Covers:** LLM-as-judge, position/verbosity/self-preference bias, inter-rater agreement, red-teaming, designing capability evals.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Agreement statistics | mandatory · code | `0-agreement.py` |
| Position-bias statistics | mandatory · code | `1-position_bias.py` |
| Judge reliability experiment | mandatory · code | `2-judge_experiment.py`, `results.json` |
| Design and run your own capability eval | mandatory · written | `written/3-capability_eval.md` |
| Automated red-teaming | advanced · code | `100-red_team.py` |

**Targets it has to hit**

- 2 automated Python tests against reference implementations and closed-form answers
- `n_pairs` ≥ 100 (Judge reliability experiment), read from `results.json`
- `kappa_judge_vs_human` ≥ -1 (Judge reliability experiment), read from `results.json`

**What I will be able to do**

- Compute Cohen's κ and explain why it corrects raw agreement for chance
- Measure position bias of an LLM judge by swapping order
- Validate a judge against your own human labels
- Design and run an eval for a capability your model plausibly has

**Key concepts:** Pairwise vs single-answer grading · Position, verbosity and self-preference bias · Order swapping and consistency · Cohen's κ vs raw agreement · Designing an eval: capability definition, item writing, grading rubric, validity · Red-teaming: manual and model-generated attacks

<details><summary>Rules and resources</summary>

Rules:

- NumPy / standard library only for tasks 0–1
- Any open LLM (local or API) as judge

Resources:

- [Zheng et al. 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Perez et al. 2022 — Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
- [Ganguli et al. 2022 — Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858)
- [Wang et al. 2023 — Large Language Models are not Fair Evaluators (position bias)](https://arxiv.org/abs/2305.17926)
- [Chiang et al. 2024 — Chatbot Arena](https://arxiv.org/abs/2403.04132)

</details>

### 0x07-chinchilla — Compute-Optimal Training: Chinchilla

**2 days** · EVAL Scaling laws in depth

**Covers:** Compute-optimal training, Chinchilla approaches 1–3, C ≈ 6ND, IsoFLOP curves, over-training for inference.

**What gets built**

| Task | Type | Deliverables |
|---|---|---|
| Compute-optimal allocation solver | mandatory · code | `0-chinchilla.py` |
| Two-page Chinchilla summary | mandatory · written | — |
| Were your A43 runs compute-optimal? | mandatory · written | `written/2-my_runs.md` |
| Fit Approach 3 on synthetic data | advanced · code | `100-fit_approach3.py` |

**Targets it has to hit**

- 1 automated Python test against reference implementations and closed-form answers

**What I will be able to do**

- Derive C ≈ 6ND from forward/backward FLOP counting
- Solve the constrained minimization of L(N, D) subject to 6ND = C in closed form
- Compute the compute-optimal model size and tokens for a given budget
- Explain the three Chinchilla estimation approaches and why they disagree slightly

**Key concepts:** Training FLOPs C ≈ 6ND · Parametric loss L(N, D) = E + A/N^α + B/D^β · Compute-optimal frontier N_opt ∝ C^a, D_opt ∝ C^b · ≈ 20 tokens per parameter rule of thumb · Over-training smaller models for cheaper inference

<details><summary>Rules and resources</summary>

Rules:

- NumPy only

Resources:

- [Hoffmann et al. 2022 — Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Besiroglu et al. 2024 — Chinchilla Scaling: A Replication Attempt](https://arxiv.org/abs/2404.10102)
- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Sardana et al. 2023 — Beyond Chinchilla-Optimal: Accounting for Inference](https://arxiv.org/abs/2401.00448)

</details>

## Part of a larger programme

This repository is one of seven that together cover a Stanford-curriculum machine learning programme, which I am working through alongside a BSc in Applied Mathematics and Computer Science at the University of Lagos:

- [ML Foundations](https://github.com/Olagold-hackxx/ml-foundations): Classical machine learning implemented from scratch in NumPy, from least squares to value iteration.
- [ML Deep Learning](https://github.com/Olagold-hackxx/ml-deep-learning): Deep learning from layer internals to a deployed vision application.
- [ML NLP](https://github.com/Olagold-hackxx/ml-nlp): Sequence models and language models, from word2vec to retrieval-augmented agents.
- [ML Generative Models & RL](https://github.com/Olagold-hackxx/ml-generative-rl): Generative models and reinforcement learning, from PixelCNN and DDPM to PPO, RLHF and DPO.
- [ML Frontier](https://github.com/Olagold-hackxx/ml-frontier) (this repository): Language models from scratch: tokenizers, architectures, training systems, scaling laws, post-training and evaluation.
- [ML Capstone](https://github.com/Olagold-hackxx/ml-capstone): A research-grade capstone project: proposal, milestone and final report.
- [ML Theory](https://github.com/Olagold-hackxx/ml-theory): Statistical learning theory with full proofs, from PAC learning to double descent.

Author: [Abdulqoyyum Aileru](https://github.com/Olagold-hackxx) · [LinkedIn](https://www.linkedin.com/in/abdulqoyyum-aileru/)
