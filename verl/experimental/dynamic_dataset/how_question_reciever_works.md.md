`question_reciever.py` pulls from the submitter **on demand** inside the `/questions` handler:

1. When `/questions?n=...` is called, it checks the in‑memory queue.
2. If there aren’t enough items, `_get_questions_with_refill()` calls `_fetch_from_submitter()`, which does:
   - `GET $QUESTION_SUBMITTER_URL?n=<needed>&split=<split>`
3. If that returns questions, it enqueues them and returns.
4. If it fails or returns none, it sleeps `QUESTION_SUBMITTER_POLL_S` and retries until `QUESTION_QUEUE_WAIT_S` expires (or forever if set to `-1`).

So polling happens **only when the dataloader requests `/questions`**, not on a background timer.

Key env vars:
- `QUESTION_SUBMITTER_URL` (required to enable polling)
- `QUESTION_SUBMITTER_POLL_S` (retry interval, default 1s)
- `QUESTION_SUBMITTER_TIMEOUT_S` (HTTP timeout, default 10s)
- `QUESTION_QUEUE_WAIT_S` (how long to wait overall, default `-1` forever)

If you want it to prefetch in the background instead, I can add a background thread to keep the queue warm.