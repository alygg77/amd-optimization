# AMD fleet brief: maya-ds4 on one AMD Instinct MI350X / MI355X (gfx950, CDNA4)

Goal: >= 100 tok/s per user at 8 concurrent users on ONE GPU, abliterated DeepSeek-V4.1-Flash-1017 (Abiray
GSQ-RCO IQ2_XXS GGUF, 131 GB weights all in VRAM, Engram 110 GB in CPU RAM). We want SUPER big gains: rewrite kernels
and structures, not flag sweeps.

## Where we are (2026-10-30 17:00 UTC)

- Base = git repo `amd/engine` commit 3439491 (our CUDA engine fork wt-engine2 4307301 + HIP guards + gfx942/gfx950 in
  the wave64 list). HIP wave64 compat: a CUDA warp is a logical 32-lane half of the 64-lane wave (shfl width 32,
  ballot shifted); kernels are CUDA-shaped. Native wave64 rewrites are allowed and welcome.
- MI350X base, flags `DS4_HEAD_COLS=16 DS4_DECODE_TRIM=4 DS4_MULTI_IDX=1 DS4_MULTI_CMP=1 DS4_WIN_GROUP_PAR=1 DS4_A1=1`:
  1 user 70.1 tok/s (TPOT 14.3 ms); 8 users 33.9 tok/s/user, 269 agg, TPOT 29.5 ms. MI355X: 73.7 / 35.5.
- Floor at 8 users: ~22 GB read per step (distinct experts ~46/layer x 40 x 8.6 MB + 6.2 GB dense) -> ~3 ms at
  ~7 TB/s. We are ~10x over the floor.

## Time map: MI355X, 8 users, rocprofv3 kernel trace (/root/prof/m355c8 on the 355), ~86 steps, GPU busy 89%

Per step (ms) at T=8 rows (kernel: share of GPU time, calls, avg us):
- Dense GEMVs ~10 ms: `mv_kernel<13,16>` (Q5_K, 16 cols) 16.4% 136/step 34.8 us = 4.7 ms; `mv_kernel<14,16>` (Q6_K)
  5.4% 38.9 us = 1.6 ms; `gemv_f32_rows_kernel` (router, F32) 5.8% 41.6 us/layer = 1.7 ms; `q8_mv_kernel<8,true>`
  4.3% 18.5 us = 1.2 ms; `q8_quant_jobs_kernel` 3.9% 244/step 4.6 us = 1.1 ms; `bf16_mv_kernel` 1.4%.
  These read a few MB each: at 7 TB/s they should take ~1-3 us, not 35-40 us.
- Routed experts ~7.7 ms: `down_kernel<29..>` (IQ1_M) 7.4% 75.9 us; `gate_up_kernel<29..>` 6.8% 123 us;
  `gate_up_kernel<16..>` (IQ2_XXS) 6.3% 106 us; `gate_up_kernel<17..>` (IQ2_XS) 2.9% 117 us; `down_kernel<16..>` 2.0%;
  `down_kernel<17..>` 1.1%. Floor for the distinct experts ~2.2 ms.
- Attention ~3.5 ms: `sparse_attn_kernel<true,0>` 6.9% 49.6 us/layer = 2.0 ms; `ext_to_ring_runs` 2.2%;
  `index_score_rows` 0.7%; `compress_kernel`; `ring_to_ext_runs`; `unpack_sel_rows`.
- Small glue ~5 ms: `hc_mix` 3.4%, `route_publish` 3.0% (21.7 us!), `finish_rows` 2.6%, `hc_pre_norm` 2.4%,
  `hc_split` 1.8%, `moe_win_group_par` 1.7%, `__amd_rocclr_copyBuffer` 1.3% (80/step), `hc_post`, `q8k_quant`,
  `finish_kernel`, `mx_combine`, `swiglu`; `sampler_greedy` 1.3% (48 us!). ~1730 kernels per step; 11% GPU idle.

## Machines and tools (Mac paths under amd)

- Boxes: `./dx.sh 350 'CMD'` = MI350X (main), `./dx.sh 355 'CMD'` = MI355X. root, 24 CPU, 251 GB RAM, ROCm 7.14
  (/opt/rocm/core-7.14, compiler /opt/rocm/llvm/bin/clang++). Same chip family: a gfx950 binary runs on both.
- Your code: the git worktree `amd/fleet/<you>` (branch `<you>`). Edit there, commit small steps on your branch.
- `./wsync.sh <you> 350|355`: rsync your worktree to the box `/root/w/<you>` and build `maya-ds4` there (-j 12).
- On the box: `bash /root/wrun.sh <you> q` = exactness check (8 prompts at once, greedy, 160 tokens, warm prompt
  cache) against `/root/res/q-ref.json` (the base binary); prints `QCHECK same N/8`. Warm A/A of the base = 8/8.
  `bash /root/wrun.sh <you> b` = the same + bench 1 and 8 users x3 (512 tokens); lines go to /root/res/summary.txt.
  Add engine env after the mode: `bash /root/wrun.sh <you> b DS4_MY_FLAG=1`.
- GPU lock: EVERY GPU job (server, fixture, profiler) runs through `/root/gpu_run.sh NAME MAXMIN CMD...`
  (wrun.sh does this for you). Compile and edit outside the lock. Keep lock time short; one server needs ~60 s to load.
- Kernel fixtures: write standalone HIP microbenchmarks (hipcc --offload-arch=gfx950) under /root/w/<you>/fx/ with
  the real tensor shapes and bytes (you can read real weights from the GGUFs in /root/model via the offsets in
  /root/model/maya-ds4/tensors.txt). Time with hipEvents, many reps, flush L2 (MI350X L2 4 MB + 256 MB Infinity
  Cache: read a >512 MB buffer between reps). Compare bits with the old kernel.
- Profile: `rocprofv3 --kernel-trace --stats -d DIR -o run --output-format csv -- <cmd>` works (ROCm 7.14).
  Engine phase timer DS4_PHASE_TIMES does not print on this multi-row path (unknown why).

## Rules

- Exactness: prove it at KERNEL level: your fixture compares the new kernel's output bits with the old kernel's on
  real weights and real-shaped inputs (T = 1, 2, 8, 16 rows); report max abs diff and the count of differing values.
  WARNING (17:06): the end-to-end QCHECK is NOT deterministic across server processes even for the base binary
  (base vs base 1-3/8; within one process 8/8). Agent `foxtrot` hunts the cause. Until it is fixed, use QCHECK only
  as a sanity check (coherent text, no NaN/garbage), not as the exactness gate.
- Aim for bit-exact kernels. A non-exact change is allowed only if it is a large gain;
  then say so plainly and keep it behind its own env flag (default off) so we can run the quality evals.
- Every new kernel/path behind an env flag `DS4_<NAME>=1` (read with the engine's ds4_env helper), default off, so
  the base stays the reference. CUDA paths must stay unchanged (`#if defined(STRATA_USE_HIP)` style as in the file).
- One structure per worker: stay in your target files/kernels; do not refactor unrelated code. Another worker owns
  each other area (see the assignments in your prompt). If you need a change outside your area, write it in RESULT.md.
- Synthetic prompts only. Never print secrets. Do not touch /root/model, /root/maya, other workers' /root/w/*, or
  tmux sessions you did not start. Never delete files you did not create. Never run git push.
- Measure honestly: report the raw numbers and paths; a gain counts only if it beats the A/A noise (8-user per-user
  tok/s varies ~0.2 between warm reps; rep 1 is always slower - use reps 2-3).

## Deliverable (every worker)

`RESULT.md` in your worktree, updated as you go: what you changed (files, kernels, flag), fixture numbers (old vs new
us, GB/s), `QCHECK` line, bench lines (1u/8u x3) with the flag on vs the base, and the next idea. Commit your work on
your branch. Final message: <= 15 lines, numbers first.

## Updates (append-only; re-read this section now and then)
- 17:15 from amd-port's earlier trace (MI355X, 8 users): idle gaps 4.93 ms/step, of which gaps >= 50 us = 3.92 ms/step
  (~5 per step, host waits, cause not yet attributed). On wave64 the engine refuses or turns off HC_FUSE, ROUTE_FUSE,
  DECODE_TRIM bits 1/2, HC_POSTNORM, HC_WIN and STEP_GRAPH (echo: find why STEP_GRAPH is refused before you build on it).
  amd-port's scripts: amd/kmap.py (kernel-class time map), amd/kan.py, amd/prof_wrap.sh (engine under rocprofv3 as the
  server exe). Prompt chunks > 32 rows crashed in sparse_attn_packed_kernel on CDNA; fixed in base (HIP guard t < kWin).
- 17:55 DETERMINISTIC GATE (foxtrot): the cross-process differences come from the CHUNKED PREFILL path (prompts of
  38-56 tokens go through forward_chunks / hipBLAS / MMQ; layer-4 residual differs ~1e-6 between processes), NOT from
  decode or batching. With DS4_PROMPT_PICK=length:100000 (all prompts through windows of 16) 3 fresh processes give
  8/8 identical. USE `bash /root/wrun.sh <you> q2` (cold + warm pass vs /root/res/q2-ref.json, script /root/qcheck2.py)
  as the end-to-end exactness gate for decode-path work. Box 350 has it; same on 355 (if missing there, copy from 350).
- 18:00 new workers: golf = clean-room decode-layer prototype (fixture only, no engine edits: how fast can ONE layer at
  T=8 run on gfx950 if written from scratch?); hotel = the big host waits (gaps >= 50 us, ~3.9 ms/step: Engram RAM
  reads/memcpy, host syncs, router/route_publish host round trips, request scheduling). echo keeps HIP graphs + kernel
  fusion; hotel owns host-side waits. Talk through RESULT.md files if you touch a shared function.
- 18:20 amd-port DSpark (MI355X, verified): on 1u 91.0 tok/s (+25%), 8u 33.9 = no gain (off 34.3). Reason: one extra
  verify row costs ~2-2.5 ms per step (rows 8->13: 29.2 -> 42.5 ms), because our GEMV/expert kernels scale with rows.
  So report for your kernels the cost at T = 8, 9, 10, 12, 16 rows, not only T=8: flat cost per extra row is what lets
  speculative decoding multiply the 8-user speed.
- 18:40 RESEARCH SYNTHESIS (workflow, 5 researchers; full JSON amd/research_synthesis.json, sources inside). Decision:
  keep maya-ds4, lift designs/kernels. Pointers per worker (all estimates, not measurements; licenses MIT/Apache):
  * alpha (dense): decode each Q5_K/Q6_K weight block ONCE for all 8 columns. Designs: llama.cpp MMQ (LDS tile, int8
    MFMA, J=8; PRs ggml-org/llama.cpp#23227, #21570) or FlyDSL "skinny" (one WG per N slab, full K, K waves through LDS
    rings, one LDS reduce, no split-K atomics; ROCm/aiter#6427, #6101: MI355X 4608x8192 M=1 8.1 us). Est. -3..-4.5 ms.
  * bravo (experts): AITER MXFP4 fused MoE for the exact DSV4.1 shape (ROCm/aiter#5852: E=384 topk=6, 8 tokens 78.9 us
    at I=1152, ~5.4 TB/s derived) as a DESIGN template: inline sort+quant+zero (#4443, #6165), sort-free direct decode
    for 2-8 tokens (#5973), group by unique expert, gate/up kernel zeroes the down output. Keep our IQ bytes: repack
    (no value change) into wave64 tiles, scales next to quants, grid tables in LDS. Est. 7.7 -> ~3-3.5 ms + 0.4-0.6 glue.
  * echo (graphs/fusion): AITER Mega-MHC HIP kernel csrc/kernels/mega_mhc.cu (ROCm/aiter#6520, MIT; 8.44 us at 8 tokens
    H=5120) can replace our hc pre/post + sinkhorn + rmsnorm at the 80 seams (est. -1..-3 ms). Whole-step HIP graph per
    batch size with routing/positions/KV pointers/top-k in fixed device buffers; do NOT call hipGraphExecKernelNodeSetParams
    per token (leaks kernarg VRAM; ROCm/rocm-systems#14059, ROCm/ATOM#2479). GPU_MAX_HW_QUEUES=1 being A/B tested by me.
  * hotel (host waits): Engram lookup (layers 1, 14) indices depend only on token ids: start hash + IQ4_NL gather on a
    side stream at step start (vLLM does this; their kernels are Triton, write ours in HIP). Measure it first.
  * charlie (router): fuse router GEMV + sqrt(softplus)+bias top-6 of 384 (aiter topk_softplus, ROCm/aiter#2995;
    vLLM wave64 fast path vllm-project/vllm#64208). Est. small unless router > 6 us/layer (ours ~40 us: big).
  * delta (attention): AITER Gluon sparse_mla for DSV4.1 shapes exists (gfx950/attention/sparse_mla.py) but needs
    Triton/Gluon AOT; use as a design reference only.
  * golf (prototype): per-layer persistent kernels, 2 launches per layer (K1: hc seam + attention front; K2: attention
    back + FFN seam + router + MoE); known gfx950 hazards: a one-row launch that did not complete and a fence that spun
    forever (ROCm/ATOM#2483).
  * Checked by me: the wave64 compat intrinsics are NOT a slow path (byte_perm = v_perm, vsub4/vcmpne4 SWAR).
  Rejected: switching base (no engine fits V4.1 on one 288 GB GPU; llama.cpp slower), MXFP4/int4 repack of experts
  (2x bytes, does not fit), ik_llama kernels (WARP_SIZE 32 hardcoded, fails on wave64), full-model megakernel (months).
