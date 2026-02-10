# PLAN E2E: Real + Synth Dual GRPO (Phased)

**Goal**
Design and validate a dual-loop GRPO system where:
- A solver loop consumes batches of 16 (14 synthetic + 2 duplicated real).
- A questioner loop generates 14 synthetic questions conditioned on 1 real question.
- A receiver orchestrates ordering, avoids race conditions, and routes rewards.

**Scope for Phase 1**
Implement and validate the questioner loop first, independent of full orchestration.

---

**Open Questions to Resolve Before Full Implementation**
- Batch semantics: When synth count < 14, do we always fill to 16 with real duplicates, or retry generation?
- Endpoint schema: Do we keep current `question_receiver.py` endpoints or add new ones for the new flow?
- Reward payload: What fields are currently returned (reward, difficulty, metadata), and do we need all fields for questioner reward?
- Prompt template: Where should the “generate a question based on this real QA” prompt live (config vs code)?
- Blocking policy: Should receiver and datasets block indefinitely or use timeouts and retries?
- Logging: Where should `question_origin` and `synthetic_valid` be logged (jsonl, wandb, receiver logs)?
- Rollout count: Confirm `actor_rollout_ref.rollout.n=14` is supported with current config.
- Naming: Confirm `dynamicgen_dataset_*` vs `dynamicgan_dataset_*` filenames (use a single naming convention).

---

**Key Assumptions**
- Data loader blocking is safe in your training loop.
- A single receiver process can serialize both loops without becoming a bottleneck.
- The questioner can produce parseable outputs at a high enough rate.
- You can split rewards between real and synthetic without changing solver reward computation.
- The system can tolerate duplicated real questions for batch padding.

---

**Phase 1 Experiments: Questioner-Only Validation**
1. **Rollout count = 14**
   - Set `actor_rollout_ref.rollout.n=14` and run a tiny training step.
   - Expected: 14 rollouts produced without errors.
2. **Parseability rate**
   - Log how often the questioner produces an unparseable question or answer.
   - Expected: low enough to handle via fallback or retry policy.
3. **Receiver handshake sanity**
   - Use a temporary stub receiver that always returns one fixed real QA pair.
   - Expected: questioner generates 14 outputs and posts them back.
4. **Latency tolerance**
   - Add artificial delay in the stub receiver.
   - Expected: training slows but does not deadlock.

---

**Phase 1: Questioner Implementation Plan**
1. Implement `dynamicgen_dataset_synth.py` to request exactly one real QA and block until available.
2. Implement `http_gsm8k_datagen_synth.py` to:
   - Build the system prompt using the real QA.
   - Generate 14 outputs per request.
   - Parse and post questions to the receiver.
3. Add explicit logging for parse errors and `question_origin=synthetic`.

---

**Phase 1: Commands to Test (Questioner Only)**
Assumes a stub receiver on `http://127.0.0.1:8080/questions` that returns one real QA and accepts synthetic posts.

```bash
# questioner test only
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_single_gpu_synth_questioner
export CUDA_VISIBLE_DEVICES=1
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    trainer.total_training_steps=20 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=0 \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_synth' \
    data.custom_cls.name='DynamicGenDataset' \
    data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_synth' \
    data.datagen.name='HttpQuestionGeneratorV3' \
    +data.datagen.skip_base_dataset=true \
    +data.datagen.url='http://127.0.0.1:8080/questions' \
    +data.datagen.n_per_batch=1 \
    +data.datagen.timeout_s=10 \
    +data.datagen.split=train \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=14 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=14 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k-dlgpu' \
    trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 | tee ${QUESTION_EXPERIMENT_NAME}.log
```

---

**Phase 2: Full Integration Plan (Revised)**
1. Implement `question_receiver_real_synth.py` with a strict state machine and per-batch IDs.
2. Implement solver-side `dynamicgen_dataset_v3.py` and `http_gsm8k_datagen_v3.py`:
   - Block until a full batch of 16 is available.
   - Pass solver rewards and difficulty back to receiver.
3. Implement receiver orchestration:
   - Receive solver batch request.
   - Request 1 real question from submitter.
   - Provide real QA to questioner loop.
   - Receive 14 synthetic questions.
   - Compose 16 (14 synth + 2 real duplicates) and send to solver.
   - Receive solver rewards and difficulty.
   - Forward synth rewards to questioner.
   - Wait for questioner ACK and then release solver for next batch.
4. Add fallback policy for invalid synth outputs (retry or duplicate real).
5. Add logging fields: `question_origin`, `synthetic_valid`, `batch_id`, `phase`.

---

**Phase 2 Experiments (Integration)**
1. End-to-end loop with small `total_training_steps=20` on both loops.
2. Inject artificial latency in receiver to ensure no deadlocks.
3. Force invalid synthetic output to test fallback policy.
4. Confirm reward separation: solver rewards passed to questioner only for synth items.

---

**Deliverables for Codex After Phase 1**
- Confirmation that questioner can reliably generate 14 parseable questions.
- Parse failure rate and chosen fallback policy.
- Finalized receiver endpoint schema and batch/ACK protocol.
- Decision on logging location and required fields.
