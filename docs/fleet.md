# dsx fleet: areas, rules, gates (2026-10-31)

## Areas (one Codex worker each, own git branch, edits only its area)

| Worker | Area |
|---|---|
| lima | routed-expert kernels (gate/up, down, expert activation prep), all batch sizes |
| oscar | hyper-connections (HC) and the sparse attention path (compressor, indexer, top-k, attention, KV) |
| november | scheduler and API: row packer (one step for all users), slots, KV/context plan, admission, head, router |
| papa | speculative decoding: native MTP/DSpark drafts, prompt lookup, batch-aware policy, lossless verify |
| quebec | dense GEMV/GEMM: attention projections, shared experts, router, head; BF16 twins vs FP8; hipBLASLt |
| romeo | launch structure and host: graphs, launch count, host gaps, uploads, Engram fetch, sampler |
| tango | integrator: merges verified flags, runs all gates, measures 1/8/16/32/64/128 users, owns the serve script |

## Rules

- Every GPU job goes through the GPU lock queue, at most ~10 minutes per job.
- Every new path sits behind a `DSXA_` environment flag, default off, until it passes the gates.
- Abliterated (heretic) weights for every speed and quality number; the original export is only the KL reference.
- The engine binary is named by absolute path, never by a generic environment variable; its md5 and commit go into
  every result directory. Builds come only from committed trees.
- Raw logs are committed; a claim without a raw file does not count.

## Gates

1. Kernel fixture on real tensors: bit-exact vs the old kernel where the math is unchanged (compare split partials or
   fixed-order sums, not atomic totals); otherwise report max error.
2. Greedy output: at least 2 reference A/A pairs in the same reservation; the candidate must fall within the A/A range
   (one-sided); all answers coherent. Becomes exact identity once the deterministic mode lands.
3. Quality: KL top-256 within +0.005 and top-1 within -0.005 of the unchanged heretic control from the same reservation.
4. Speed: base / candidate / base / candidate interleaved, warm repetitions, distinct prompts or the API load tool,
   every batch size reported including regressions; no heuristic chosen on the timed data.
