steps_to_run_gsm05b_remote_real_and_synth:

```bash
# questioner
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_single_gpu_synth_questioner
export CUDA_VISIBLE_DEVICES=1
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    trainer.total_training_steps=1000 \
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
    trainer.log_val_generations=3 \
    trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
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
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k-dlgpu' \
    trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 | tee ${QUESTION_EXPERIMENT_NAME}.log

```

```bash
# SOLVER
export CUDA_VISIBLE_DEVICES=0
export QUESTION_EXPERIMENT_NAME=qwen2_0.5b_single_gpu_real_synth_solver 
PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    trainer.total_training_steps=1000 \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts_workers=0 \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset_v3' \
    data.custom_cls.name='DynamicGenDataset' \
    data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen_v3' \
    data.datagen.name='HttpQuestionGeneratorV3' \
    +data.datagen.skip_base_dataset=true \
    +data.datagen.url='http://127.0.0.1:8080/questions' \
    +data.datagen.n_per_batch=16 \
    +data.datagen.timeout_s=10 \
    +data.datagen.split=train \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    trainer.log_val_generations=3 \
    trainer.rollout_data_dir=/workspace/verl/verl/${QUESTION_EXPERIMENT_NAME} \
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
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k-dlgpu' \
    trainer.experiment_name=${QUESTION_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 | tee ${QUESTION_EXPERIMENT_NAME}.log

```