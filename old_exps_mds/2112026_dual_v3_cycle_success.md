# Dual Loop V3 Cycle Success Guide (N=2)

## Goal
Increase cycle finalization rate and reduce `cycle_failed` so the solver consistently receives exactly 2 parsable questions per cycle.

## Primary Failure Modes Seen
1. `cycle_failed` dominates because questioner parse success is low.
2. Questioner crashes with `DynamicGenDataset received an empty filtered snapshot` when generated prompts exceed `data.max_prompt_length`.
3. Solver waits with 504 if fewer than required (`n=2`) questions are eligible in a cycle.

## Required Configuration Changes
Use these settings together (not partially):

### Questioner
- `actor_rollout_ref.rollout.n=2`
- `+custom_reward_function.reward_kwargs.expected_cycle_size=2`
- `+custom_reward_function.reward_kwargs.min_required_per_cycle=2`
- `data.max_prompt_length=1024` (important: avoid empty filtered snapshot)
- `+data.datagen.snapshot_size=1`
- `+data.datagen.n_per_batch=1`
- `+data.datagen.refresh_on_batch_end=true`
- `+custom_reward_function.reward_kwargs.cycle_timeout_s=120`

### Solver
- `data.train_batch_size=2`
- `actor_rollout_ref.actor.ppo_mini_batch_size=2`
- `+data.datagen.n_per_batch=2`
- `+data.datagen.snapshot_size=2`
- `+data.datagen.wait_s=20`
- `+data.datagen.timeout_s=25`
- `+data.datagen.refresh_on_batch_end=true`

## Launch Order
1. Receiver
2. Questioner
3. Solver

## Receiver Command
```bash
cd /workspace/verl/verl
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_dual_v5_cycle_n2
export DUAL_LOOP_FAIL_ON_TIMEOUT=true
export DUAL_LOOP_CYCLE_TIMEOUT_S=180
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=true
export DUAL_LOOP_TRACE_MAX_TEXT_CHARS=220

python3 -m verl.experimental.dynamic_dataset.question_reciever_v5_solver_questioner \
  --host 127.0.0.1 \
  --port 8080
```

## Questioner Command (Cycle-Gated N=2)
```bash
cd /workspace/verl/verl
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_dual_questioner_v5_n2
export CUDA_VISIBLE_DEVICES=1
export DUAL_LOOP_COORDINATOR_URL='http://127.0.0.1:8080'
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=max_split_size_mb:64
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=true
export DUAL_LOOP_TRACE_MAX_TEXT_CHARS=220

CUDA_VISIBLE_DEVICES=1 RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_questioner_v5 PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=0 \
  data.train_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v5_synth' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v5_synth' \
  data.datagen.name='HttpQuestionGeneratorV5Synth' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url='http://127.0.0.1:8080' \
  +data.datagen.snapshot_size=1 \
  +data.datagen.n_per_batch=1 \
  +data.datagen.wait_s=5 \
  +data.datagen.split=train \
  +data.datagen.refresh_on_batch_end=true \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v5_synth' \
  custom_reward_function.name='compute_synth_difficulty_score' \
  +custom_reward_function.reward_kwargs.coordinator_url='http://127.0.0.1:8080' \
  +custom_reward_function.reward_kwargs.submit_timeout_s=20 \
  +custom_reward_function.reward_kwargs.wait_timeout_s=1200 \
  +custom_reward_function.reward_kwargs.poll_interval_s=0.5 \
  +custom_reward_function.reward_kwargs.require_numeric_final_answer=true \
  +custom_reward_function.reward_kwargs.difficulty_target=0.5 \
  +custom_reward_function.reward_kwargs.w_parse=0.2 \
  +custom_reward_function.reward_kwargs.w_cycle_parse=0.4 \
  +custom_reward_function.reward_kwargs.w_diff=0.8 \
  +custom_reward_function.reward_kwargs.cycle_bonus=0.25 \
  +custom_reward_function.reward_kwargs.difficulty_target_band=0.1 \
  +custom_reward_function.reward_kwargs.fail_on_timeout=true \
  +custom_reward_function.reward_kwargs.expected_cycle_size=2 \
  +custom_reward_function.reward_kwargs.min_required_per_cycle=2 \
  +custom_reward_function.reward_kwargs.cycle_timeout_s=120 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
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
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.rollout.tensor_model_parallel_size=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_dual_loop' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

## Solver Command (N=2 intake)
```bash
cd /workspace/verl/verl
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_dual_solver_v5_n2
export CUDA_VISIBLE_DEVICES=0
export DUAL_LOOP_COORDINATOR_URL='http://127.0.0.1:8080'
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=max_split_size_mb:64
export DUAL_LOOP_TRACE=true
export DUAL_LOOP_TRACE_INCLUDE_TEXT=true
export DUAL_LOOP_TRACE_MAX_TEXT_CHARS=220

CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local RAY_TMPDIR=/tmp/ray_solver_v5 PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=0 \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v5_solver' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v5_solver' \
  data.datagen.name='HttpQuestionGeneratorV5Solver' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url='http://127.0.0.1:8080' \
  +data.datagen.snapshot_size=2 \
  +data.datagen.n_per_batch=2 \
  +data.datagen.timeout_s=25 \
  +data.datagen.wait_s=20 \
  +data.datagen.split=train \
  +data.datagen.strict_response_post=false \
  +data.datagen.refresh_on_batch_end=true \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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
  actor_rollout_ref.rollout.n=10 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.rollout.tensor_model_parallel_size=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_dual_loop' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

## Monitoring
```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/debug/state
```

Cycle health should trend toward:
- `cycle_counts.finalized` increasing steadily
- `cycle_counts.failed` no longer dominating
- `candidate_counts.solved` increasing with lower `cycle_failed`

## Triage Rules
1. If questioner crashes with empty snapshot:
- Raise `data.max_prompt_length` further (e.g. 1536).
- Keep `data.filter_overlong_prompts=True`.

2. If solver repeatedly gets 504 on `/solver/questions`:
- Check `/debug/state` for `solver_queue_size` and `cycle_counts.failed`.
- If failed cycles dominate, parse quality is still too low.

3. If cycles finalize but solver stays `difficulty=0`:
- Solver reward quality/model performance issue (not queueing deadlock).
- Keep queue settings fixed and tune solver separately.
