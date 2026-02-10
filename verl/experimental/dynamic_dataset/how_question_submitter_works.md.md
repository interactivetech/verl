`question_submitter.py` supports two modes: push and serve.

Push mode (default)
- Loads GSM8K and builds shuffled indices per split.
- Builds a batch of size `n_per_batch` and extracts `solution` from the GSM8K `answer` using the `#### <number>` pattern.
- POSTs to the receiver at `QUESTION_RECEIVER_URL` (`/submit`).
- Sleeps `delay_s` seconds and repeats until `max_batches` (if set).

Serve mode (`--serve`)
- Starts a FastAPI server that exposes `GET /questions?n=...&split=...`.
- Returns a fresh batch directly (no receiver involved).
- Used by `question_reciever.py` for on‑demand pull‑through.

Key CLI flags
- `--receiver-url`, `--n-per-batch`, `--delay-s`, `--max-batches`
- `--serve`, `--host`, `--port`

Key env vars
- `QUESTION_RECEIVER_URL` (push target)
- `GSM8K_DATASET`, `GSM8K_SUBSET`, `GSM8K_SPLIT`, `GSM8K_LOCAL_PATH`
- `SEED`
