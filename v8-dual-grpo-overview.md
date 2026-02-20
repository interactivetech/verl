# V8 Dual GRPO Overview

## TLDR
1. Install environment and container workflow: `v8-dual-grpo-runbook-full-1000.md`.
2. Run quick wiring check first: `v8-dual-grpo-runbook-smoke.md`.
3. Run full training: `v8-dual-grpo-runbook-full-1000.md`.
4. Debug issues: `v8-dual-grpo-troubleshooting.md`.

## Problem Framing
V8 runs a dual-loop training setup:
- Questioner generates candidate math problems.
- Solver attempts those problems and reports per-candidate difficulty.
- Questioner reward is derived from solver feedback.

This creates an online curriculum loop where question quality and solver behavior co-evolve.

## System Components
- Coordinator API service:
  `verl/experimental/dynamic_dataset/question_reciever_v8_solver_questioner.py`
- Questioner datagen + reward:
  `verl/experimental/dynamic_dataset/http_gsm8k_v8_questioner.py`
- Solver datagen + reward + result post:
  `verl/experimental/dynamic_dataset/http_gsm8k_v8_solver.py`
- Dataset entrypoints (v8 aliases to v3 dynamic dataset):
  `verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_questioner.py`
  `verl/experimental/dynamic_dataset/dynamicgen_dataset_v8_solver.py`

## End-to-End Dataflow
1. Questioner rollout emits text expected to contain `<question>...</question>` and `\boxed{...}`.
2. Questioner reward parses output, builds candidate records, and sends one batch to:
   `POST /questioner/submit_and_wait`.
3. Coordinator registers candidates and queues them.
4. Solver datagen pulls `n` candidates from:
   `GET /solver/questions`.
5. Solver runs rollout; batch-end hook aggregates majority agreement and posts:
   `POST /solver/results`.
6. Coordinator marks candidates resolved and wakes waiting questioner request.
7. Questioner reward finalizes scores:
   - base uncertainty reward for parse-valid queueable candidates
   - invalid parse penalty for parse failures
   - optional diversity penalty subtraction for parse-valid rows

## API Contract (Coordinator)
| Endpoint | Method | Called by | Purpose | Typical failure |
|---|---|---|---|---|
| `/questioner/submit_and_wait` | POST | questioner reward | register candidates, block for solver results | `504 solver_result_timeout` |
| `/solver/questions` | GET | solver datagen | fetch `n` queued candidates | `504 solver_fetch_timeout` |
| `/solver/results` | POST | solver on_batch_end | publish per-candidate difficulty/result stats | `404 unknown candidate_id` |
| `/health` | GET | ops | quick liveness + queue counters | n/a |
| `/debug/state` | GET | ops | queue state + sample candidate metadata | n/a |

## Reward Semantics
### Questioner
- Parse-valid + queueable candidate score uses uncertainty form:
  `1 - 2 * abs(p_hat - 0.5)` where `p_hat` comes from solver-reported difficulty.
- Invalid parse gets `invalid_parse_penalty` (default in baseline: `-1.0`).
- Diversity mechanism is single-path in v8:
  - toggle: `enable_diversity_penalty`
  - params: `diversity_penalty_weight`, `diversity_distance_threshold`, `diversity_linkage`

### Solver
- Rollout responses grouped per `candidate_id`.
- Majority agreement ratio becomes `difficulty`.
- Self-consistency reward uses majority-match with `invalid_score` fallback.

## Important Runtime Notes
- Solver fetch can wait indefinitely when `wait_s=-1`, with retry loop around 503/504.
- Questioner can be configured to fail fast on timeout (`fail_on_timeout=true`) or fallback score.
- Dynamic dataset is bounded snapshot replacement, not append growth.
- Validation rows with data source `openai/gsm8k` are scored locally and do not hit coordinator.
