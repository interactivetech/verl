# Multi-Round Self-Play Plan (v9)

## Goal
Implement a live dual-loop variant where each questioner candidate is solved across multiple solver training rounds (e.g., 8-15 rounds) before the candidate is marked resolved and returned to the questioner as final reward.

This keeps online co-training, but changes candidate lifecycle from:
- v8: `submit -> one solver round -> resolved`
to:
- v9: `submit -> multiple solver rounds with aggregation -> resolved`

## New Files To Create
1. `02052026_project/verl/verl/experimental/dynamic_dataset/question_reciever_v9_solver_questioner.py`
- New receiver/coordinator with lease + multi-round state machine.

2. `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v9_solver.py`
- Solver datagen + post-results client updated to support round-aware posting and aggregate-friendly payloads.

3. `02052026_project/verl/verl/experimental/dynamic_dataset/http_gsm8k_v9_questioner.py`
- Questioner reward client updated to submit candidates with optional per-batch round config and consume final aggregated solver stats.

4. `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v9_solver.py`
- Alias wrapper for v9 solver datagen module path isolation.

5. `02052026_project/verl/verl/experimental/dynamic_dataset/dynamicgen_dataset_v9_questioner.py`
- Alias wrapper for v9 questioner datagen module path isolation.

6. `02052026_project/verl/2192026_v9_multi_round_q306_commands.md`
- New runbook markdown with copy-paste commands for receiver, questioner, solver, restart/cleanup, health checks, and ablations.

## What Will Be Implemented

### 1) Receiver: multi-round candidate lifecycle
In `question_reciever_v9_solver_questioner.py`:
- Candidate state fields:
  - `target_rounds`, `rounds_done`, `status`, `resolved`
  - lease metadata: `lease_owner`, `lease_until`, `last_assigned_at`
  - aggregate stats: answer histogram / parsed counts / total rollouts
- Assignment behavior:
  - Serve unresolved candidates even if previously assigned, when lease expired or completed prior round.
  - Fair scheduling so old unresolved candidates are prioritized.
- Resolution behavior:
  - On `/solver/results`, increment `rounds_done` and merge round stats.
  - Resolve only when `rounds_done >= target_rounds`.
  - Final difficulty/reward signal computed from aggregated stats across all rounds.
- Safety:
  - Lease timeout -> requeue.
  - Idempotency guard for duplicate round posts (per `candidate_id + round_id`).
  - Backpressure limits for unresolved candidates.

### 2) Solver posting: round-aware payload
In `http_gsm8k_v9_solver.py`:
- Include `round_id` (or monotonic local step token) in posted results.
- Post richer stats for aggregation robustness:
  - `n_total`, parsed/unparsed counts
  - answer-count histogram (preferred) or majority + support counts
- Keep current majority-vote reward training path for solver GRPO batch reward unless explicitly changed.

### 3) Questioner reward wait path
In `http_gsm8k_v9_questioner.py`:
- Submit candidates with optional target rounds (batch-level override supported).
- Continue batch `submit_and_wait`, but now wait for final resolution after multi-round aggregation.
- Keep current parse/validity handling and diversity penalty toggles as-is.

### 4) Config surface (env + command flags)
Expose v9 knobs (receiver and clients):
- `MULTIROUND_TARGET_ROUNDS` (default 1 for backward compatibility)
- `MULTIROUND_LEASE_TTL_S`
- `MULTIROUND_MAX_UNRESOLVED`
- `MULTIROUND_REQUEUE_ON_LEASE_EXPIRY=true/false`
- `MULTIROUND_STRICT_IDEMPOTENCY=true/false`

### 5) Command runbook generation (required deliverable)
Create `02052026_project/verl/2192026_v9_multi_round_q306_commands.md` with:
- Receiver start command (`question_reciever_v9_solver_questioner`).
- Questioner training command (v9 questioner modules and experiment names).
- Solver training command (v9 solver modules and experiment names).
- Health/debug commands (`/health`, `/debug/state`) and expected key fields.
- Safe restart and cleanup commands (receiver + ray tmpdir/process reset guidance).
- Ablation command variants for `target_rounds=1/8/12/15`.

## Efficient Implementation Strategy
1. Copy v8 files into v9 files first (no behavior change).
2. Add feature-flagged multi-round logic with default `target_rounds=1`.
3. Validate parity with v8 at `target_rounds=1`.
4. Enable `target_rounds=8` and test tiny runs.
5. Scale to 12/15 rounds after queue health and latency checks.

## Validation and Debug Plan
Add/verify debug endpoints include:
- unresolved counts, resolved counts
- average `rounds_done/target_rounds`
- lease expiries, requeues, duplicate-post drops
- queue depth split by status

Run checks:
1. `target_rounds=1` reproduces v8 behavior.
2. `target_rounds=8` shows questioner waits longer but no deadlock.
3. Solver continuously receives work for unresolved candidates.
4. No stuck `assigned` candidates after solver restarts.

## Risks / Tradeoffs
1. Latency increase for questioner rewards.
2. Higher receiver memory pressure from unresolved state.
3. More coupling between solver uptime and questioner throughput.
4. Need strict lease/requeue handling to avoid deadlocks.
5. Potential solver overfitting to active unresolved pool without fairness controls.

## Questions To Resolve Ambiguity
1. Target rounds policy:
- Resolved: configurable (global default + optional per-batch/per-candidate override).

2. Aggregation source of truth:
- Resolved: configurable mode.
- Supported modes:
  - `aggregate_all_rounds`: final reward from aggregated majority ratio across all rounds.
  - `final_round_only`: final reward from last round majority ratio only.

3. Round granularity:
- Resolved: one round per solver train step.
- Receiver/accounting implementation note: a round is counted when the solver successfully posts that step's `/solver/results`.

4. Duplicate handling:
- Resolved: strict idempotency (ignore duplicate `(candidate_id, round_id)` posts).

5. Timeout policy:
- Resolved: fail hard if target rounds are not completed in time (no partial/fallback reward).

6. Fairness policy:
- Resolved default: unresolved-first scheduling.
- No mixed ratio policy in first v9 implementation.

7. Solver-side training behavior:
- Resolved: keep current solver reward unchanged (batch majority each step).

8. Capacity limits:
- Resolved: keep a safety cap for unresolved candidates to prevent unbounded memory growth when solver stalls.
- Implementation default: enabled with configurable high threshold (`MULTIROUND_MAX_UNRESOLVED`), tunable/raiseable.

## Suggested First Experiment After v9
1. `target_rounds=1` (parity control)
2. `target_rounds=8` (your intended mode)
3. `target_rounds=12` (stress test)
Track:
- deadlocks/timeouts
- queue depth stability
- solver validation trend
- questioner reward variance
