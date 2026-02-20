# 2026-02-11 Questioner Pretrain Commands

This is the question-only pretrain stack (not dual-loop).  
It uses:
- submitter: `question_submitter_v2`
- receiver: `question_reciever_v2_synth`
- trainer: `dynamicgen_dataset_v3_synth` + `http_gsm8k_datagen_v3_synth`

Run in 3 terminals.

## 1) Submitter
```bash
cd /workspace/verl/verl

python3 -m verl.experimental.dynamic_dataset.question_submitter_v2 \
  --serve \
  --host 127.0.0.1 \
  --port 8090
```

## 2) Receiver (question-only)
```bash
cd /workspace/verl/verl

export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_questioner_pretrain_v1
export QUESTION_SUBMITTER_URL='http://127.0.0.1:8090/questions'
export QUESTION_ASKED_LOG_MODE=all

python3 -m verl.experimental.dynamic_dataset.question_reciever_v2_synth \
  --host 127.0.0.1 \
  --port 8080
```

## 3) Questioner Pretrain (Stage A: parse bootstrap)
```bash
cd /workspace/verl/verl

export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_questioner_pretrain_v1
export CUDA_VISIBLE_DEVICES=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=max_split_size_mb:64

PYTHONUNBUFFERED=1 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=1000 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=0 \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v3_synth' \
  data.custom_cls.name='DynamicGenDataset' \
  data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3_synth' \
  data.datagen.name='HttpQuestionGeneratorV3Synth' \
  +data.datagen.skip_base_dataset=true \
  +data.datagen.url='http://127.0.0.1:8080/questions' \
  +data.datagen.snapshot_size=32 \
  +data.datagen.n_per_batch=1 \
  +data.datagen.timeout_s=10 \
  +data.datagen.split=train \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3_synth' \
  custom_reward_function.name='compute_synth_qa_score' \
  +custom_reward_function.reward_kwargs.require_numeric_final_answer=false \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.log_val_generations=3 \
  trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_questioner_pretrain' \
  trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=50 \
  | tee ${QUESTION_EXPERIMENT_NAME}.log
```

## 4) Optional Stage B (numeric-final hardening)
Rerun command above with this change:
```bash
+custom_reward_function.reward_kwargs.require_numeric_final_answer=true
```

## 5) Verify Services During Pretrain
```bash
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8080/health
curl -s "http://127.0.0.1:8090/questions?n=2&split=train" | jq .
curl -s "http://127.0.0.1:8080/questions?n=2&split=train" | jq .
```

## 6) Use Pretrained Weights in End-to-End Dual Loop
Questioner should warm-start from pretrain checkpoint.

### 6.1 Find latest pretrain checkpoint
```bash
cd /workspace/verl/verl
ls -1d checkpoints/verl_grpo_questioner_pretrain/qwen2_0.5b_questioner_pretrain_v1/global_step_* | sort -V | tail -n 1
```

### 6.2 Add to dual-loop questioner command
Use the returned path as `CKPT_PATH`:
```bash
trainer.resume_mode=resume_path \
trainer.resume_from_path=${CKPT_PATH} \
```

Example:
```bash
trainer.resume_mode=resume_path \
trainer.resume_from_path=/workspace/verl/verl/checkpoints/verl_grpo_questioner_pretrain/qwen2_0.5b_questioner_pretrain_v1/global_step_400 \
```

Notes:
- Resume path must include `global_step_<N>`.
- Keep model family unchanged when warm-starting (same Qwen 0.5B base).
- Usually only the dual-loop questioner uses this warm-start; solver can remain from base unless you also pretrain solver separately.
