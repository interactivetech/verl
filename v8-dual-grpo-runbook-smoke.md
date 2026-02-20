# V8 Dual GRPO Runbook: Smoke Test

## Goal
Fast end-to-end wiring check for the v8 dual loop:
- coordinator up
- solver fetch/post path works
- questioner submit/wait path works
- W&B logging works

This profile is not for quality benchmarking.

## Profile
- Single node, 2 GPUs minimum.
- `trainer.total_training_steps=20` for solver and questioner.
- Reduced batch and rollout counts for speed.

## 1) Environment Install (Apptainer + Slurm Interactive)
Choose one pinned image tag (default used below):

```bash
export V8_DOCKER_IMAGE=verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
export V8_SIF=$HOME/containers/v8_dual_grpo.sif
mkdir -p "$(dirname "$V8_SIF")"
apptainer pull "$V8_SIF" "docker://$V8_DOCKER_IMAGE"
```

Start an interactive Slurm allocation (adjust to your cluster):

```bash
salloc --nodes=1 --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=08:00:00
```

Open a container shell on each terminal/pane:

```bash
export REPO_HOST=/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/2026_projects/02052026_project/verl
apptainer shell --nv -B "${REPO_HOST}:/workspace/verl" "$V8_SIF"
```

Inside container:

```bash
cd /workspace/verl
pip3 install --no-deps -e .
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

W&B:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
wandb login --relogin "$WANDB_API_KEY"
export WANDB_ENTITY=<your_wandb_entity>
```

Common run vars (run once per shell):

```bash
cd /workspace/verl
export MODEL_SHORT=qwen3_0.6b
export TS=$(date +%Y%m%d_%H%M)
export WB_PROJECT=verl_grpo_v8_dual_loop_q306b
export EXP_SOLVER="${MODEL_SHORT}_v8_dual_solver_smoke_b2_n2_${TS}"
export EXP_QUESTIONER="${MODEL_SHORT}_v8_dual_questioner_smoke_b2_n2_${TS}"
```

## 2) Terminal 1: Coordinator
```bash
cd /workspace/verl

DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
DUAL_LOOP_TRACE_MAX_TEXT_CHARS=200 \
SOLVER_FETCH_WAIT_S=-1 \
QUESTIONER_RESULT_WAIT_S=-1 \
python3 -m verl.experimental.dynamic_dataset.question_reciever_v8_solver_questioner \
  --host 127.0.0.1 \
  --port 8080
```

## 3) Terminal 2: Solver (Smoke)
```bash
cd /workspace/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=1 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_solver_v8_smoke \
PYTHONUNBUFFERED=1 \
WANDB_ENTITY=${WANDB_ENTITY} \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v8_solver' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_solver' \
  data.datagen.name='HttpQuestionGeneratorV8Solver' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.coordinator_url='http://127.0.0.1:8080' \
  +data.datagen.snapshot_size=2 \
  +data.datagen.n_per_batch=2 \
  +data.datagen.max_question_chars=1200 \
  +data.datagen.wait_s=-1 \
  +data.datagen.timeout_s=30 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_solver' \
  custom_reward_function.name='compute_solver_self_consistency_scores_batch' \
  +custom_reward_function.reward_kwargs.invalid_score=-1.0 \
  reward_manager.name=batch \
  reward_model.use_reward_loop=false \
  actor_rollout_ref.model.path='Qwen/Qwen3-0.6B' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir="/workspace/verl/${EXP_SOLVER}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${EXP_SOLVER}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=20 \
  trainer.total_epochs=20 \
  | tee "${EXP_SOLVER}.log"
```

## 4) Terminal 3: Questioner (Smoke)
```bash
cd /workspace/verl

unset PYTORCH_CUDA_ALLOC_CONF
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
DUAL_LOOP_TRACE=true \
DUAL_LOOP_TRACE_INCLUDE_TEXT=false \
CUDA_VISIBLE_DEVICES=0 \
RAY_ADDRESS=local \
RAY_TMPDIR=/tmp/ray_questioner_v8_smoke \
PYTHONUNBUFFERED=1 \
WANDB_ENTITY=${WANDB_ENTITY} \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v8_questioner' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_questioner' \
  data.datagen.name='HttpQuestionGeneratorV8Questioner' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.snapshot_size=2 \
  +data.datagen.n_per_batch=2 \
  +data.datagen.refresh_on_batch_end=true \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_v8_questioner' \
  custom_reward_function.name='compute_synth_difficulty_scores_batch' \
  +custom_reward_function.reward_kwargs.coordinator_url='http://127.0.0.1:8080' \
  +custom_reward_function.reward_kwargs.submit_timeout_s=72000 \
  +custom_reward_function.reward_kwargs.wait_timeout_s=600 \
  +custom_reward_function.reward_kwargs.fail_on_timeout=true \
  +custom_reward_function.reward_kwargs.require_numeric_ground_truth=true \
  +custom_reward_function.reward_kwargs.invalid_parse_penalty=-1.0 \
  +custom_reward_function.reward_kwargs.enable_diversity_penalty=true \
  +custom_reward_function.reward_kwargs.diversity_penalty_weight=1.0 \
  +custom_reward_function.reward_kwargs.diversity_distance_threshold=0.5 \
  +custom_reward_function.reward_kwargs.diversity_linkage='average' \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  reward_manager.name=batch \
  actor_rollout_ref.model.path='Qwen/Qwen3-0.6B' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=0 \
  trainer.rollout_data_dir="/workspace/verl/${EXP_QUESTIONER}" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WB_PROJECT}" \
  trainer.experiment_name="${EXP_QUESTIONER}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  trainer.save_freq=20 \
  trainer.test_freq=1000000 \
  trainer.val_before_train=false \
  trainer.total_training_steps=20 \
  trainer.total_epochs=20 \
  | tee "${EXP_QUESTIONER}.log"
```

## 5) Health Checks
```bash
curl -sS --max-time 2 http://127.0.0.1:8080/health
curl -sS --max-time 2 http://127.0.0.1:8080/debug/state
```

Expected signals:
- `solver_queue_size` changes over time (not permanently stuck at one value).
- Solver log shows fetch and post cycles.
- Questioner log shows submit-and-wait returning resolved results.
- W&B has two runs with expected experiment names.

## 6) Exit Criteria
- No persistent 504 timeouts on both data paths.
- Both training jobs pass at least one optimization step.
- Coordinator endpoints remain responsive.

