# 2112 Pretrain Questioner Penalize Length

## Is this a good idea?
Yes, with one caveat.
- Good: it fights collapse to very short/easy questions.
- Caveat: too much length pressure can cause verbose but low-quality questions.

In the current codebase, there is no true per-seed length reward term yet in `compute_synth_qa_score`.
So this experiment uses a **runnable proxy**:
1. Prompt instruction to stay near seed-question length.
2. Reward gating via `min_question_chars` (short questions get 0 reward).
3. Keep parse/numeric-final requirements on.

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

export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_questioner_pretrain_lenproxy_v1
export QUESTION_SUBMITTER_URL='http://127.0.0.1:8090/questions'
export QUESTION_ASKED_LOG_MODE=all

python3 -m verl.experimental.dynamic_dataset.question_reciever_v2_synth \
  --host 127.0.0.1 \
  --port 8080
```

## 3) Questioner Pretrain (Length-Penalty Proxy)
```bash
cd /workspace/verl/verl

export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_questioner_pretrain_lenproxy_v1
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
  +data.datagen.user_prompt_template='You are given one real seed question and its answer. Seed Question: {seed_question} Seed Answer: {seed_answer} Task: Generate ONE new, different math word problem inspired by the seed. IMPORTANT: keep the generated question length close to the seed question length (roughly 0.8x to 1.2x in characters). Output valid JSON only with keys question and answer. The answer must end with #### <final_number>.' \
  custom_reward_function.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3_synth' \
  custom_reward_function.name='compute_synth_qa_score' \
  +custom_reward_function.reward_kwargs.min_question_chars=120 \
  +custom_reward_function.reward_kwargs.min_answer_chars=20 \
  +custom_reward_function.reward_kwargs.require_numeric_final_answer=true \
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

## 4) Monitor Quality
```bash
# Parse metrics in trainer log
rg -n "parse_ok|score/mean|numeric_final_ok" qwen2_0.5b_questioner_pretrain_lenproxy_v1.log | tail -n 40

# Inspect generated rollouts
ls -1 qwen2_0.5b_questioner_pretrain_lenproxy_v1 | tail -n 10
```

## 5) Recommended Acceptance Criteria
- `parse_ok` stays high and stable.
- `question` text no longer collapses to very short templates.
- No sharp drop in diversity across rollout files.

## Note on True Per-Seed Length Penalty
This file uses a runnable proxy. A true reward term like
`1 - abs(len(synth_q)-len(seed_q))/len(seed_q)`
requires a small reward-function extension (not present yet in `compute_synth_qa_score`).
