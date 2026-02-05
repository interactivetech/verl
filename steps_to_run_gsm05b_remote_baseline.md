verl remote gsm8k dataloader test

This baseline proves out training a Qwen 2.5 0.5B model from a remote http server
that handles serving questions from the dataset. 

This is a proof of concept to develop a dual GRPO meta learning loop. 

```
docker exec -it 80d2cf065f36 /bin/bash
apt-get update && apt-get install -y python3-venv python3-pip

python3.10 -m venv remote-venv

conda deactivate
source remote-venv/bin/activate
pip install fastapi uvicorn datasets
<!-- cd /workspace/verl/verl/experimental/dynamic_dataset -->
cd verl/experimental/dynamic_dataset/
uvicorn gsm8k_server:app --host 0.0.0.0 --port 8080
OR 
uvicorn verl.experimental.dynamic_dataset.gsm8k_server:app --host 0.0.0.0 --port 8080 --workers 2

curl -s "http://127.0.0.1:8080/questions?n=16&split=train"

create empty dataset:
```
That error happens because Dataset.from_list([]) can’t reconcile explicit features with an empty list. Use from_dict with empty columns (or create one dummy row then drop it).

Here’s a clean, working snippet (and note the stray PYint(out_path)ut_path) in your run was a paste glitch):

python3 - <<'PY'
from datasets import Dataset, Features, Value, Sequence
import os

features = Features({
    "data_source": Value("string"),
    "prompt": Sequence({"role": Value("string"), "content": Value("string")}),
    "ability": Value("string"),
    "reward_model": {
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    "extra_info": {
        "split": Value("string"),
        "index": Value("int64"),
        "answer": Value("string"),
        "question": Value("string"),
    },
})

empty = {
    "data_source": [],
    "prompt": [],
    "ability": [],
    "reward_model": [],
    "extra_info": [],
}

ds = Dataset.from_dict(empty, features=features)

out_path = "/workspace/verl/empty_train.parquet"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
ds.to_parquet(out_path)
print(out_path)
PY
```


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
    data.custom_cls.path='pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset' \
    data.custom_cls.name='DynamicGenDataset' \
    data.datagen.path='pkg://verl.experimental.dynamic_dataset.http_gsm8k_datagen' \
    data.datagen.name='HttpQuestionGenerator' \
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
    trainer.rollout_data_dir=/workspace/verl/verl/rollout_data_0.5b-remote_v4 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k-dlgpu' \
    trainer.experiment_name='qwen2_0.5b_single_gpu_remote_v4' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 | tee qwen2.5_0.5b_grpo_1gpu_remote_v4.log


```