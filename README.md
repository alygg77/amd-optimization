# AMD optimization: DeepSeek-V4.1-Flash decode on one AMD Instinct GPU

Status: 2026-10-31 (UTC). This page is the live summary of the work. History is at the end.

## Goal

Serve an **abliterated DeepSeek-V4.1-Flash-1017** as fast as possible on **one** AMD gfx950 GPU (MI350X / MI355X,
288 GB HBM), with up to **1M tokens of context** per user.

| Users | What we optimize | Target |
|---|---|---|
| 8 | speed per user | 100 tok/s per user |
| 16, 32 | speed per user | as high as possible |
| 64, 128 | total throughput | as high as possible |
| any | context length | up to 1M tokens per user |

## Engine: dsx-amd

We moved from our Maya fork to **dsx** ([lxe/dsx](https://github.com/lxe/dsx), MIT, Rust host + HIP kernels, ~18K lines
for the AMD path). dsx was built for this model only. Our worker ported it from gfx942 (MI325X) to gfx950 (native OCP
FP8) and fixed a CB3 scale bug on the way (CB3 experts decoded at 2x scale on gfx950; all numbers before that fix are
invalid).

- Weights: [lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-mxfp4)
  (270 GB). Dense FP8 is byte-identical to the official checkpoint; routed experts are 59% MXFP4 and 41% 3-bit
  codebook (CB3). Published quality: KL 0.049, top-1 90.6%.
- Abliteration without re-quantization: the dense FP8 tensors are byte-identical, so we copy the changed tensors of a
  published abliterated FP8 checkpoint into the dsx shards (sha256 checked, manifest per build):
  - **heretic** (pqhaz): 38 `attn.wo_b` tensors. All speed work uses this build.
  - **mrfakename**: 124 tensors (`wo_b` + `shared_experts.w2`).
  - The final pick follows a refusal / KL / coherence A/B (running).
- Heretic quality on dsx (same reservation, 8184 scored positions): ppl 8.794, KL 0.0649, top-1 0.8954
  (original: 8.704 / 0.0495 / 0.9087). New changes are judged against the heretic control, not the original.

## Where we are (heretic weights, API load, greedy, 256-512 output tokens)

| Box | 1 user | 8 users | 16 users | 32 users | 64 users total | 128 users total |
|---|---|---|---|---|---|---|
| MI355X | 115.4 tok/s | 43.5 tok/s/user | 32.3 | 16.4 | 527 tok/s | 557 tok/s |
| MI350X | ~105 | 40.5 | 30.0 | 15.3 | 493 | 523 |
| Maya fork (MI350X; 8u with best flags, 16/32u base) | 70.1 | 35.3 | 20.0 | 11.8 | - | - |

dsx is 1.6-2.2x faster than our Maya fork at every width, before any of the work below.

## Where the time goes

- **8 users: 21-23 ms per step.** Realistic byte floor ~6 ms (measured read ceiling ~6.3 TB/s; real expert union with
  distinct prompts). Routed experts ~8.9 ms; small glue kernels: 240 `prep_t` launches ~1.7 ms; ~1050 launches per step.
- **The batched path has a fixed extra cost:** at B=1 it takes 14.2 ms vs 8.2 ms in the single-user path; the
  non-expert chain costs ~11.5 vs ~5.7 ms. Every 8-user step pays this.
- **32-128 users:** the server splits decode users into groups of 28 (`runner.rs`), so 128 users read all weights
  5 times per round. Kernel-only sweeps reach ~1,350 tok/s at 128 rows vs 557 through the API.
- **Measurement trap we found:** the stock batch sweep copies one prompt into every slot, so the expert union is too
  small and times are optimistic. All claims now use distinct prompts or the API.

## Leads (not yet integrated; each must pass the gates below)

| Change | Owner | Measured / predicted |
|---|---|---|
| Expert tile mode (LDS-staged activations + scales), bit-exact partials | lima | **8 users 44.5 -> 52.5 tok/s/user (+18%)** on MI355X, +12.6% on MI350X (original weights, order-biased; heretic A/B running) |
| Row packer: one decode step for all users (remove the 28-user groups) | november | predicted 32u +30%, 128u 557 -> 750-815 tok/s; more with batched admission |
| Port the single-user step structure to the batched path (fused norm/quant into products, grouped launches) | quebec + oscar | up to -5.8 ms at 8 users (upper bound) |
| Dense kernels for 17-256 rows (LDS-staged MX-FP8 GEMM), head GEMM | quebec | fixture: -1.0 ms at 8u, -3.2 ms at 32u, -6.2 ms at 128u; head 720 -> 436 us at B=8 |
| Persistent expert kernel over a task list, no split-K atomics, fused SwiGLU/combine | lima | predicted +13-17% at 8 users |
| Fused sparse attention + hyper-connection seams | oscar | predicted ~-3 ms at 8u, ~-8 ms at 128u |
| Speculative decoding by batch size (native MTP/DSpark, lookup for code) | papa | predicted +20-30% per user at 8u on code, ~0 on prose |
| Plain decode skips the verify/commit path; async next step; Engram off the step start | november, romeo | predicted +1-6% each |
| Data-parallel replicas on 2 GPUs + router (deployment, no code) | - | ~2x at 16-128 users |
| Deterministic mode with "pair buffers" (each sum gets at most 2 adds onto +0, order-free) | all | < 0.5% cost; makes every quality gate exact |

Lossy options (fewer experts per token, colder experts at lower bits) are listed but held until the owner approves them.

## Bugs found by our reviews (being fixed)

- Host-to-device copies on the legacy null stream race with queued kernels on the batched DSpark path.
- DSpark drafting asserts `t <= 256` rows: with the row packer, more than 51 drafting users would kill every request.
- A parked slot's history can run past the stop token under speculation.
- The server defaults to temperature 1.0 while GPU sampling is off by default: one sampled client slows its group.
- Speculative verify rows can pick a different kernel family than plain rows (chosen by total rows, not by users).

## 1M context

The model supports a 1M-token context. Work in progress:
- memory plan per slot vs context length (KV ring, compressed KV, indexer keys) and how many users x context fit in the
  ~37 GB left after the weights (and after BF16 twins / DSpark);
- decode speed vs context (the indexer scores grow with context; the selected top-k does not), measured at 4K, 32K,
  128K and then 1M;
- fixes for the parts that grow with context (indexer scoring, top-k, block skipping).

## How we work

- **Workers:** 7 Codex agents (gpt-6.1-sol, extra-high reasoning, priority mode off), one area each, one git branch each:
  experts, HC + attention, scheduler/API, speculative decoding, dense kernels, launch/host, and an integrator that
  merges only verified flags into one branch and measures 1/8/16/32/64/128 users.
- **Research workflows** (Claude Opus + Haiku, ~7 agents each, with adversarial checks): code gap audit, external
  kernel sweep, structure design panel, byte reduction, determinism map, speculative decoding lab, free expert review,
  live system probe, prototype race (real kernels on the GPUs), path-gap profile, long-context capacity, GPU queue
  audit, and a re-derivation of every worker claim from raw files.
- **excession** harness: 32 parallel review/implementation workers on a separate copy of the engine (CPU only;
  patches land behind default-off flags and are then tested on the GPUs).
- **Hardware:** two DigitalOcean gfx950 droplets (MI350X, MI355X), one GPU each, shared through a GPU lock queue.
- **Gates for every gain:** abliterated (heretic) weights for every arm; base / candidate / base / candidate in one
  reservation; binary md5 recorded; distinct prompts; greedy output within the measured A/A range (until the
  deterministic mode lands); KL and top-1 within +0.005 / -0.005 of the heretic control; every batch size reported,
  including regressions.

## Scaling to more GPUs later

Almost everything transfers: data-parallel replicas (one full model per GPU + router) need no code and multiply total
throughput; a tensor/expert-parallel split would raise per-user speed further but needs cross-GPU communication code.

## History

- 3x A100 80 GB, tensor parallel 3, Maya engine: 36.8 tok/s/user at 8 users.
- One MI350X/MI355X, Maya fork ported to HIP wave64: 35.5 tok/s/user at 8 users (+4-6% from our kernel flags).
  Correction: an earlier "DSpark +25% at 1 user" table ran a worker's experimental binary, not the base; withdrawn.
- llama.cpp on MI325X: 18.4 tok/s/user at 8 users. orcarouter Mix-128G: faster in llama.cpp but 20% worse perplexity.
- 2026-10-30: dsx found by a research sweep, ported to gfx950, made the main engine; Maya work stopped.

## Docs

- docs/fleet.md: worker areas, rules and gates.
- docs/research-2026-10-31.md: what each research workflow found.
- docs/research-synthesis.md: the earlier (Maya-era) design research.
