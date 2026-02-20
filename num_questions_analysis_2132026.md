# Synthetic-vs-Real Question Analysis (Qwen3-0.6B Dual Loop)

Date: 2026-02-13

## Scope and data used

I analyzed local artifacts for:
- Questioner runs: `q1`, `q4`, `q16`
  - `02052026_project/verl/qwen3_0.6b_dual_questioner_v3_n1_val_e2`
  - `02052026_project/verl/qwen3_0.6b_dual_questioner_v3_n4_val`
  - `02052026_project/verl/qwen3_0.6b_dual_questioner_v3_n16_val`
- Solver runs:
  - Synthetic loop: `02052026_project/verl/qwen3_0.6b_dual_solver_v3_n1_val_e2`
  - Real-question baseline: `02052026_project/verl/rollout_data_q3-0.6b-single2`

All main metrics below are for the first 100 steps where available.
- `q1`: full 100-step window available.
- `q4`: only 35 steps available in this run.
- `q16`: only 15 steps available in this run.

## Executive findings

1. Your intuition is correct: `q1` had by far the least zero-reward collapse.
- Questioner zero-reward rate:
  - `q1`: 28%
  - `q4`: 100%
  - `q16`: 100%

2. The main failure mode for `q4/q16` was parseability + strict cycle gating.
- `min_required_per_cycle = expected_cycle_size = n` means all candidates must be valid to avoid cycle failure.
- Observed parse rates made this nearly impossible for larger `n`.

3. Synthetic questions underperform real questions for solver learning.
- Solver mean@1 (strict score) first 100 steps:
  - Real baseline: `0.6401`
  - Synthetic `q1`: `0.3340`
- Even with relaxed numeric matching (ignore strict format issues):
  - Real: `0.7048`
  - Synthetic `q1`: `0.4520`

4. `q1` likely succeeded because it reduced combinatorial cycle failure and produced non-zero learning signal early.
- It did not eliminate failure, but it made training viable.

## Question structuredness: q1 vs q4 vs q16

### Questioner metrics (first 100 steps where available)

| Run | Rows | Steps | parse_ok | question_ok | answer_ok | numeric_final_ok | cycle_failed | selected | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 (`n=1`) | 100 | 100 | 0.720 | 0.880 | 0.880 | 0.720 | 0.280 | 0.720 | 0.443 |
| q4 (`n=4`) | 140 | 35 | 0.107 | 0.157 | 0.150 | 0.107 | 1.000 | 0.000 | 0.000 |
| q16 (`n=16`) | 240 | 15 | 0.242 | 0.425 | 0.425 | 0.242 | 1.000 | 0.000 | 0.000 |

### Cycle parseability explains collapse

Cycle parsable counts:
- `q4`: 22 cycles had `0/4` parsable, 11 had `1/4`, 2 had `2/4`, none had `4/4`.
- `q16`: cycles ranged `2/16` to `8/16`, none had `16/16`.
- `q1`: 72 cycles had `1/1`, 28 had `0/1`.

If per-candidate parse probability is `p`, all-parse probability is `p^n`:
- `q1`: `0.72^1 = 0.72`
- `q4`: `0.107^4 = 0.00013`
- `q16`: `0.242^16 ≈ 1.36e-10`

This is exactly the observed behavior: `q4/q16` almost always cycle-fail regardless of occasional valid candidates.

## Why q1 improved: evidence from run behavior

- `q1` first 100 steps: 72 solved cycles, 28 failed cycles.
- Reward decomposition in `q1`:
  - `parse_ok=1` rows mean reward `0.6159`
  - `parse_ok=0` rows mean reward `0.0`
- Trend in `q1` first 100:
  - steps 1-20: parse 0.60, reward 0.309
  - steps 21-60: parse 0.675, reward 0.441
  - steps 61-100: parse 0.825, reward 0.514

So `q1` gave enough positive samples to learn formatting and question production, while `q4/q16` stayed in all-zero territory.

## Synthetic vs real questions for solver

### Solver first 100-step comparison

| Dataset | Rows | Strict mean score | Zero-score rate |
|---|---:|---:|---:|
| Real baseline | 16000 | 0.6401 | 0.3599 |
| Synthetic q1 | 1000 | 0.3340 | 0.6660 |

Relaxed numeric match (extract final number regardless of strict `####` format):
- Real: `0.7048`
- Synthetic q1: `0.4520`

Conclusion: this is not only a strict-format issue. Synthetic data quality is lower for solver training.

### Concrete synthetic-question issues observed

1. Format/answer-style mismatch still causes avoidable zero scores.
- Example output format: `#### $15.00 per day.` with ground truth `15` scored `0.0`.

2. Some generated prompts are awkward/underspecified/oddly phrased.
- Example: "got across" phrasing and tautological phrasing correlated with low scores.

3. Distribution shift from GSM8K natural style.
- Synthetic questions include unnatural constructions and variable quality answers, reducing transfer to standard eval style.

## Additional config evidence

`q1` command includes:
- `+data.apply_chat_template_kwargs.enable_thinking=False`

`q4/q16` commands did not include this.
Observed in outputs:
- `q4/q16`: 100% included `<think>`-style long preambles, high unparsed rates.
- `q1 e2`: 0% with `<think>`, substantially better parseability.

This appears to be one of the main structural improvements, in addition to `n=1`.

## Answer to your hypothesis

Your hypothesis is supported by the data:
- Yes, the dual GRPO loop depends heavily on question quality/structuredness.
- Yes, `q1` reduced cycle penalty pressure enough to keep training alive.
- Yes, `q4/q16` likely failed because parseability was too low for all-or-nothing cycle completion.

One nuance:
- `q1` did not remove penalty; it made penalty proportional/learnable instead of almost guaranteed. That is why it improved learning signal.

## Recommendations

### 1) Keep `n=1` until parse reliability is very high

Gate escalation by measured parse metrics:
- Move to `n=2` only after rolling 200-step `parse_ok >= 0.90` and `cycle_failed <= 0.10`.
- Move to `n=4` only after `parse_ok >= 0.95` at `n=2`.

### 2) Keep thinking disabled for questioner (critical)

Keep:
- `+data.apply_chat_template_kwargs.enable_thinking=False`

Also enforce output hard-format in prompt:
- "Return raw JSON object only, no markdown, no code fences, no extra keys."

### 3) Relax all-or-nothing cycle gating

Current `min_required_per_cycle = n` is too brittle.
Try:
- `min_required_per_cycle = max(1, ceil(0.5*n))`

This preserves quality pressure but prevents total reward collapse.

### 4) Add a cheap question-quality verifier reward

Add penalties for:
- non-standalone wording
- answer units/format incompatible with solver scorer
- self-contradictory asks (asking for quantity already explicitly given)

Add bonuses for:
- unambiguous target variable
- GSM8K-like concise structure
- numeric final answer format exactly aligned with scorer

### 5) Add curriculum on synthetic generation hardness

Start with easier transformations of real GSM8K:
- paraphrase + number substitution + unit-preserving edits

Then gradually allow freer generation once parse/validity is stable.

### 6) Model suggestions

- Questioner: try a slightly stronger model (e.g., Qwen3-1.7B LoRA) for structure compliance.
- Solver can stay 0.6B initially if data quality is fixed.
- If memory constrained, keep stronger model only on questioner side.

### 7) Immediate prompt update to test

Use this questioner output contract:

```
You must output exactly one JSON object with keys: question, answer.
Constraints:
- No markdown, no code fences, no extra text.
- question: one standalone GSM8K-style word problem, unambiguous.
- answer: concise derivation ending with exactly `#### <number>` (number only, no $ or units).
- The final number must answer the asked quantity exactly.
- Avoid tautologies (do not ask for a value already explicitly given).
```

## Bottom line

- `q1` is currently the only stable regime because it avoids combinatorial cycle failure.
- Synthetic questions are currently lower quality than real GSM8K prompts for solver learning.
- The next win is not bigger `n`; it is improving parseability + validity + scorer alignment, then scaling `n` gradually.
