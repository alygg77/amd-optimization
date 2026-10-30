# AMD optimization: DeepSeek-V4.1-Flash decode on one AMD Instinct GPU

Goal: **>= 100 output tok/s per user at 8 concurrent users on ONE GPU**, for an abliterated DeepSeek-V4.1-Flash-1017
(2-bit GGUF experts). This repo holds the direction, the measurements and the plan. Status: 2026-10-30.

## Why AMD, why one GPU

- The 2-bit model is 131 GB of weights (plus the 110.6 GB Engram tables, kept in CPU RAM). It fits on ONE
  MI325X / MI350X / MI355X (256-288 GB HBM). One GPU means no all-reduce and no tensor-parallel traffic.
- Before the pivot we ran it on 3x A100 80 GB with tensor parallel 3: best 36.8 tok/s/user at 8 users (exact),
  38.0 (not exact). One MI350X/MI355X already gives about the same.
- We target gfx950 (CDNA4: MI350X / MI355X). MI300X/MI325X (gfx942, CDNA3) run the same HIP code; only tuning differs.

## Model and weights

- DeepSeek-V4.1-Flash-1017: 40 layers, hidden 5120, 384 routed experts (top-6), moe_intermediate 2304,
  hyper-connections (hc_mult 4, Sinkhorn), compressed sparse attention (window 128, indexer top-512), Engram n-gram
  memory (layers 1 and 14), DSpark drafter for speculative decoding.
- Weights: Abiray `DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF` (ISTA-DASLab GSQ-RCO quant with the
  refusal direction removed): experts IQ1_M / IQ2_XXS / IQ2_XS, dense Q5_K / Q6_K / Q8_0 / BF16.
- Compared: orcarouter `DeepSeek-V4.1-Flash-1017-abliterated-GGUF` Mix-128G. It is faster in llama.cpp at 8 users
  (26.4 vs 18.3 tok/s/user) because it reads fewer bytes, but its perplexity is 20% worse (wikitext-2: 3.54 vs 2.95).
  We keep Abiray.

## Engine choice

- We use our fork of the Project Maya engine `maya-ds4` (C++/CUDA with a HIP backend). We ported it to CDNA wave64
  (the compat layer maps a CUDA warp onto half a 64-lane wavefront) and fixed the faults on the way (see Findings).
- Why not another engine: vLLM / SGLang / ATOM support V4.1 on MI3xx, but only with the official weights
  (~292 GiB without Engram), so they need 2+ GPUs. llama.cpp (PR #31086) runs the GGUF on one GPU but is slower at
  8 users (18.3 vs 33.9 tok/s/user; ~2600 kernels per token). Decision (research, docs/research-synthesis.md): keep
  maya-ds4 and lift designs from AITER / FlyDSL / llama.cpp MMQ (MIT / Apache).

## Where we are (MI350X, one GPU, greedy, 512-token answers, median of warm reps)

| Engine / setting | 1 user tok/s | 8 users tok/s per user | 8 users total |
|---|---|---|---|
| llama.cpp master, MI325X | 56.5 | 18.4 | ~140 |
| maya-ds4 HIP, defaults, MI355X | 73.6 | 29.0 | 231 |
| maya-ds4 HIP, best flags, MI350X | 70.1 | 33.9 | 269 |
| maya-ds4 HIP, best flags, MI355X | 73.7 | 35.5 | 282 |
| + DSpark speculative decoding, MI355X | **91.0** | 33.9 (no gain yet) | 268 |

## Where the time goes (8 users, one step = 29.5 ms; floor ~3 ms)

From a rocprofv3 kernel trace on MI355X (~1730 kernels per step, GPU busy 84-89%):

| Part | ms per step | Note |
|---|---|---|
| Dense GEMVs (Q5_K, Q6_K, router F32, q8, bf16) | ~10 | 35-40 us per call for 3-35 MB matrices; the byte floor is ~1-5 us each |
| Routed experts (IQ1_M / IQ2_XXS / IQ2_XS) | ~7.7 | byte floor ~2.2 ms for ~46 distinct experts per layer |
| Attention (sparse attention, indexer, ring copies) | ~3.5 | reads each KV row once per head |
| Small glue kernels (hc, finish, quantize, copies) | ~5 | ~1700 launches |
| Idle gaps | ~4.9 | ~5 host waits >= 50 us per step = 3.9 ms |

The two GPUs (MI325X 6 TB/s, MI355X 8 TB/s) give the same speed within 2-3%: we are limited by kernel efficiency and
launches, not by HBM bandwidth. So the work is software.

## Direction

1. Kernel rewrites for gfx950 (wave64-native, every weight read once for all 8 users):
   - dense Q5_K/Q6_K multi-column GEMV: decode each block once for all columns (llama.cpp MMQ or FlyDSL skinny design);
   - routed experts: one fused MoE in the style of AITER's DSV4.1 MXFP4 MoE, but on our 2-bit codebook bytes;
   - router + top-6 fused; q8 quantize steps fused into their consumers;
   - attention: share KV reads across heads, fuse the ring/unpack copies.
2. Launch structure: whole decode step as a HIP graph (fixed device buffers), AITER Mega-MHC for the hyper-connection
   seams, remove host waits (Engram gather on a side stream).
3. Speculative decoding: DSpark already gives +25% at 1 user. At 8 users it pays only when an extra verify row is
   almost free (today +1 row = +2-2.5 ms), which the kernel work above gives.
4. Later: per-layer persistent kernels (2 launches per layer). A clean-room prototype measures how far a fresh design is.

Estimate (research, not measured): steps 1-2 bring a step to ~18-20 ms (~50 tok/s/user); 100 tok/s/user needs
DSpark at 8 users and persistent kernels.

## How we work

- One git branch per worker, one target area each, Codex (gpt-6.1-sol / gpt-6-astra) and Claude agents. Fleet brief:
  docs/fleet.md. Each change ships behind an env flag (default off).
- Two gfx950 boxes (MI350X + MI355X, one GPU each) shared through a GPU lock queue.
- Exactness gate: kernel-level bit compare in fixtures, plus an end-to-end greedy check of 8 concurrent prompts
  (prompts through the decode windows, see Findings).

## Findings so far

- Engram offset fault: our pack's tensor table had Engram offsets 160 bytes too small for the Engram GGUF on HF
  (header 416 B vs 256 B), giving NaN in layer 1 and out-of-bounds expert reads. Data fix only.
- Prompt chunks > 32 rows crashed in a sparse-attention kernel on CDNA (no WMMA path): HIP guard added.
- End-to-end greedy output differs between server processes on HIP; the cause is the chunked prefill path
  (hipBLAS / MMQ), not decode or batching. With prompts through the decode windows, output is 8/8 identical.
- llama.cpp master crashes at 8 users in a hipBLASLt kernel after ~200 tokens; ROCBLAS_USE_HIPBLASLT=0 avoids it.
- The orcarouter Mix-128G build trades 20% perplexity for 25% fewer bytes: not a substitute.
- DSpark on HIP: acceptance ~78-80%, +25% at 1 user, 0% at 8 users (verify-row cost).

## Docs

- docs/research-synthesis.md: the ranked list of designs to lift, with sources and the rejected options.
- docs/fleet.md: the worker brief (areas, rules, measurement method).
