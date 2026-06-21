# Thinker local PD scheduler — A100 reproduction notes

This branch is the v3 implementation of
`sgl-project/sglang-omni#841`: the env-gated thinker local
prefill/decode scheduler scaffolding, with a measured end-to-end A100
reproduction. The hook is wired up but does not yet change scheduling
behavior; it emits events only. See "Status" and "Measurements" below.

## Status

Two earlier prototypes on this machine were withdrawn because they
crashed the thinker scheduler thread:

- **Prototype 1 (v1, this branch's commit history before 956ce0c)**:
  injected ready-decode rids into `batch.reqs` directly. Crashed in
  sglang's `filter_batch` with `IndexError: list index out of range`
  on `multimodal_inputs[i]`, because modifying `reqs` did not keep the
  parallel per-req state arrays (`multimodal_inputs`,
  `req_pool_indices`, `seq_lens`, `output_ids`, …) in sync.
- **Prototype 2 (v2 in commit `b5f6...` reverted)**: moved requests
  from `running_batch.reqs` to `self.waiting_queue` and re-admitted
  them. Crashed in sglang's `alloc_for_extend` with
  `RuntimeError: alloc_req_slots runs out of memory. req_to_token_pool.available_size()=0`,
  because the moved request had already been allocated a slot from
  `req_to_token_pool` and sglang was unaware.
- **Prototype 3 (this branch, post-`956ce0c`)**: emits the
  ready-decode admit events but does **not** move request objects
  between batches. Sglang's existing prefill→decode transition in
  `get_next_batch_to_run` continues to merge prefill output into the
  running batch. The hook is now a measurement scaffold; the actual
  scheduling change is out of scope for this PR.

The H100 numbers in #841 remain the only published measurement
showing the scheduling change itself is worth implementing on A100.

## Measurements (env on vs env off, this machine)

Qwen3-Omni-30B-A3B-Instruct, text-only rollout, max_tokens=256,
N=16 / N=32, 2 repeats each (mean reported), 1 prompt reused across
all concurrent clients, all clients released at t=0.

- Server: `examples/run_qwen3_omni_server.py` (text-only pipeline, 6
  stages: preprocessing, image_encoder, audio_encoder, mm_aggregate,
  thinker, decode).
- Driver script: `/root/gpufree-data/bench-results/text_rollout_stress.py`
  (standalone, no project dependency).
- GPU: A100-SXM4-80GB, single GPU, TP=1.
- Raw data: `bench-results/baseline-gpu1/text_rollout_stress.json`
  (env off) and `bench-results/localpd-v3-events*/text_rollout_stress.json`
  (env on, this branch).

| N | mode | p50 (s) | p95 (s) | p99 (s) | output tok/s | success |
|---:|---|---:|---:|---:|---:|---|
| 16 | baseline (env off) | 1.995 | 2.179 | 2.196 | 743.5 | 16/16 |
| 16 | local-PD events-only (env on) | 2.119 | 2.345 | 2.378 | 672.2 | 16/16 |
| 32 | baseline (env off) | 3.302 | 4.639 | 4.679 | 715.5 | 32/32 |
| 32 | local-PD events-only (env on) | 3.322 | 4.521 | 4.574 | 715.8 | 32/32 |

Differences are within run-to-run noise (≤ 10 %), as expected for an
events-only hook that does not actually alter the scheduling decision.
No scheduler crashes with `SGLANG_OMNI_ENABLE_THINKER_PD=1` on this
branch.

## What this PR ships

- `SGLANG_OMNI_ENABLE_THINKER_PD` (EnvBool, default off) and
  `SGLANG_OMNI_THINKER_PD_READY_DECODE_LIMIT` (EnvInt, default 32).
- `OmniScheduler.enable_local_pd` kwarg. Mutually exclusive with
  `enable_overlap` and `enable_async_decode` (raises `ValueError`).
- A small FIFO (`_ready_decode` of rids + `_ready_enter_ts`) and four
  helpers: enqueue-on-prefill, admit-on-cycle, abort, mutual-exclusion
  check.
- Three profiler events: `pd_ready_enter`, `pd_ready_admit`,
  `pd_ready_drop`.
- 9 unit tests under `tests/unit_test/pipeline/test_thinker_pd_scheduler.py`
  covering constructor wiring, enqueue / admit / limit-drop / abort /
  mutex / event payload.
- `bootstrap.py` wires the env flag into
  `create_thinker_scheduler` (talker factory unchanged).
- Default behavior is byte-for-byte unchanged when the env flag is
  unset (the env is `False`).

## What this PR does not do

- It does not change scheduling behavior. The events tell you
  *when* a ready-decode admission would have happened, not *that* a
  request was actually held back. To recover the H100 numbers in
  #841, the next step is to change the implementation of
  `_local_pd_admit_into_waiting_queue` to do the request transfer
  safely. Both attempts so far have failed because of two different
  batch-state invariants in sglang 0.5.8; the right fix is most
  likely a small upstream patch to `sglang.srt.scheduler.Scheduler`
  (e.g. a "prefill-to-decode hand-off" hook) that sglang-omni would
  call from `process_batch_result_prefill`. Filing that upstream
  patch is the suggested follow-up for a separate PR.

## Diff stat

```
 sglang_omni/environ.py                              |   3 +
 sglang_omni/models/qwen3_omni/bootstrap.py          |   3 +
 sglang_omni/scheduling/omni_scheduler.py            | 100 ++++++++++
 tests/unit_test/pipeline/test_thinker_pd_scheduler.py | 201 +++++++++++++++++++
 4 files changed, 307 insertions(+)
```

Refs: sgl-project/sglang-omni#841 (motivation), #842 (proposal draft).