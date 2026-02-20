# Calibration Guide: Dual-Loop GRPO

## Scope
This guide explains how to calibrate a **dual-loop GRPO** setup (questioner + solver) when dataset characteristics change.

Assumptions:
- strict fail-fast orchestration
- in-memory coordinator
- solver and questioner exchange data through coordinator endpoints

## Why Calibration Is Needed
Changing datasets changes:
- prompt length distribution
- parseability rate
- solver success rate distribution
- response length / latency
- queue pressure and timeout risk

These directly affect rollout stability, reward quality, and deadlock/timeout behavior.

## Core Calibration Knobs
### Loop orchestration
- `solver_fetch_wait_s`: wait for solver to get enough questions.
- `solver_result_wait_s`: wait for questioner to receive solver-derived rewards.
- `fail_on_timeout`: for strict mode keep `true`.

Recommended starting values:
- `solver_fetch_wait_s=120`
- `solver_result_wait_s=180`

### Questioner loop
- `rollout.n` (candidates per seed prompt)
- parse reward + difficulty reward weights:
  - `w_parse`, `w_diff`
- parseability thresholds (question/answer length, strictness)

### Solver loop
- `data.train_batch_size` (consumption rate)
- rollout settings affecting latency
- reward strictness for correctness

### Dataset/dataflow
- snapshot size
- seed question fetch size
- parse gate policy (what gets sent to solver)

## Hard-Coded Ratios to Recalibrate
When dataset changes, re-check these ratios first.

1. Candidate supply ratio:
- `questioner_supply_per_step = questioner_train_batch_size * questioner_rollout_n * parse_rate`
- must exceed solver demand with margin:
- `questioner_supply_per_step >= 1.2 * solver_train_batch_size`

2. Difficulty coverage:
- ensure generated questions span difficulty around target `0.5`.
- if solver is mostly 0 or mostly 1, difficulty reward collapses.

3. Timeout safety:
- `solver_result_wait_s > worst_case_step_time + validation_stall + jitter`
- from recent logs, use at least 180s for strict mode.

## Recommended Reward Form
Use parse-gated combined reward:

`R = parse_ok * (w_parse + w_diff * max(0, 1 - 2*abs(difficulty - 0.5)))`

Recommended schedule:
- Phase A (format learning): `w_parse=1.0`, `w_diff=0.0`
- Phase B (transition): `w_parse=0.5`, `w_diff=0.5`
- Phase C (target): `w_parse=0.2`, `w_diff=0.8`

Only move phase when parse rate is stable above threshold (e.g. >90%).

## Calibration Procedure (Dataset Change)
1. Run questioner-only sanity check for 50-100 steps.
2. Measure parse rate and strict-json rate on answered logs.
3. Estimate solver demand and required parseable supply.
4. Set orchestration timeouts using observed worst stalls.
5. Run dual-loop with small solver batch size (e.g. 4) and higher questioner rollout (e.g. 10).
6. Check timeout/fail frequency and queue depths.
7. Adjust one dimension at a time:
- first supply (`rollout.n`, parse strictness)
- then timeout
- then reward weights

## Validation Stall Handling
Simplest robust policy for strict MVP:
- keep timeouts high enough to survive validation pauses.

If validation stalls are ~45s and can overlap, use:
- `solver_fetch_wait_s=120`
- `solver_result_wait_s=180`

## Metrics to Monitor
### Questioner
- `parse_ok` rate
- strict-json rate
- question length / answer length distributions
- difficulty-target reward mean

### Solver
- accuracy / pass-rate per synthetic question
- latency per rollout batch

### Coordinator
- queue depth (pending candidates, pending solver work)
- timeout count by endpoint
- in-flight cycle count

## Failure Signatures and Fixes
### Frequent timeout failures
Likely causes:
- parseable supply too low
- validation stalls
- solver throughput too low

Fix order:
1. increase `solver_result_wait_s`
2. lower solver batch size
3. increase questioner `rollout.n`
4. relax parse formatting strictness temporarily

### Reward collapse to zero
Likely causes:
- parse failure spike
- solver always easy/hard (difficulty saturates)

Fix:
1. raise parse weight temporarily (`w_parse`)
2. inspect candidate format drift
3. widen seed diversity

### Mode collapse to trivially easy questions
Likely causes:
- difficulty component too weak
- solver too weak/too strong relative to question distribution

Fix:
1. increase `w_diff`
2. rebalance solver capacity or sampling

## Minimal Retune Checklist
- [ ] parse rate >= 90%
- [ ] strict-json trend improving
- [ ] no timeout failures for 100+ steps
- [ ] solver demand satisfied with >=20% buffer
- [ ] difficulty reward not collapsed at extremes

## Notes
- Keep strict fail-fast during calibration; do not mask timeout failures with fallback rewards.
- Move to tolerant policies only after orchestration is stable.
