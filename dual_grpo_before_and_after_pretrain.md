# Dual GRPO Before vs After Questioner Pretrain

## Compared runs
- Before pretrain warm-start:
  - Questioner: `qwen2_0.5b_dual_questioner_v5_n2`
  - Solver: `qwen2_0.5b_dual_solver_v5_n2`
  - Receiver archive: `qwen2_0.5b_dual_v5_cycle_n2_questions_asked_answered`
- After pretrain warm-start:
  - Questioner: `qwen2_0.5b_dual_questioner_v5_n2_warm`
  - Solver: `qwen2_0.5b_dual_solver_v5_n2_warm`
  - Receiver archive: `qwen2_0.5b_dual_v5_cycle_n2_warm_questions_asked_answered`

## Headline result
Pretraining the questioner is a major win in this setup. The loop moved from mostly cycle-failed/unsolved to mostly parsable/selected with materially better solver correctness.

## Core metrics

### Questioner (rollout JSONL aggregates)
- Samples:
  - Before: 2000
  - After: 322
- `parse_ok` mean:
  - Before: `0.2455`
  - After: `0.9596`
- `numeric_final_ok` mean:
  - Before: `0.2455`
  - After: `0.9596`
- `selected` mean:
  - Before: `0.0870`
  - After: `0.9317`
- `cycle_failed` mean:
  - Before: `0.9130`
  - After: `0.0683`
- `score` mean:
  - Before: `0.0405`
  - After: `0.4671`
- `difficulty` mean:
  - Before: `0.0035`
  - After: `0.2984`

Trend inside each run:
- Before run degraded (collapse pattern):
  - Early `parse_ok`: `0.3550` -> Late `0.0575`
  - Early score: `0.0643` -> Late `0.0051`
- After run stayed strong:
  - Early `parse_ok`: `1.0000` -> Late `0.9844`
  - Early score: `0.4375` -> Late `0.4812`

### Solver (rollout JSONL aggregates)
- Samples:
  - Before: 1740
  - After: 3020
- Accuracy (`acc`) mean:
  - Before: `0.0408`
  - After: `0.3248`
- Score mean:
  - Before: `0.0408`
  - After: `0.3248`
- Output format quality:
  - Has `####`: Before `0.2402`, After `0.8070`
  - Has numeric `####`: Before `0.0948`, After `0.7179`

Trend inside each run:
- Before: Early acc `0.0000` -> Late `0.1983`
- After: Early acc `0.0083` -> Late `0.4950`

### Receiver archives (asked/answered)
- Before:
  - Unique asked candidates: `174`
  - Unique answered candidates: `174`
  - Solver correct rate by `n_correct/n`: `0.0408`
  - Difficulty mean: `0.0408`
  - Answered throughput: `83.6` candidates/hour
- After:
  - Unique asked candidates: `306`
  - Unique answered candidates: `304`
  - Solver correct rate by `n_correct/n`: `0.3250`
  - Difficulty mean: `0.3250`
  - Answered throughput: `341.7` candidates/hour

Interpretation:
- End-to-end loop reliability is much higher after warm-start.
- Solver is now learning from substantially better-formed and more solvable traffic.
- Difficulty is still often low (median remains `0.0` in archives), so there is room to push richer problems.

## Recommendation on your proposed next ideas

### 1) Increase number of questions exchanged per cycle
Recommendation: **Yes, but gradually and symmetrically**.
- Move from `N=2` to `N=3` first (questioner rollout + cycle expected size + solver `n_per_batch` aligned).
- Do not jump to very high N immediately; bigger N increases chance one bad sample blocks a cycle if gating is strict.
- Add a small retry/fill mechanism for failed parses to keep cycle completion high.

### 2) Increase `train_batch_size` and `ppo_mini_batch_size`
Recommendation: **Yes, this is likely the highest short-term stability gain after pretrain**.
- Start with modest increments (e.g., questioner `1 -> 2`, solver `2 -> 4` if memory allows).
- Keep `ppo_micro_batch_size_per_gpu=1` and scale mini-batch first.
- Watch for slower wall-clock and timeout pressure; increase cycle/reward timeouts accordingly.

### 3) Larger model experiments (1.5B LoRA / 1.5B full / 4B full)
Recommendation: **Good, but second wave after batch+reward tuning**.
- 1.5B LoRA is the best cost/perf next step.
- 1.5B full can help further but is much heavier.
- 4B full without pretraining is risky on stability/cost; do after proving reward/dataflow at smaller scale.

### 4) Improve reward function
Recommendation: **Yes, very important now**.
- Current biggest gap: many accepted items still have low effective difficulty.
- Add stronger shaping for:
  - strict parse + numeric final answer,
  - target difficulty band adherence,
  - anti-triviality (length/operation-count/novelty checks),
  - solver calibration reward (not too easy, not impossible).

## Easy wins likely to give large gains
- Keep pretrain warm-start as default for questioner.
- Add curriculum on difficulty target:
  - start lower target band, then ramp to desired band.
- Add duplicate/near-duplicate filtering in receiver queue to improve diversity.
- Penalize structurally malformed outputs early and hard (before queue admission).
- Increase solver validation cadence moderately (e.g., `test_freq=50` or `100`) and tune timeouts to avoid false deadlocks.
- Track these KPIs every run in one script:
  - questioner `parse_ok`, `selected`, `cycle_failed`,
  - solver `acc`, numeric `####` rate,
  - receiver queue depth + cycle fail rate.

## Suggested next experiment order
1. Keep warm-start + tune batch sizes up one step.
2. Add reward shaping for difficulty-band and anti-triviality.
3. Raise cycle size from 2 to 3 with strict alignment.
4. Then run 1.5B LoRA variant with the same tuned reward/flow.
