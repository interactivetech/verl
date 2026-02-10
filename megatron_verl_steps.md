# Steps to install Megatron backend to verl:

# Updated Megatron + veRL LoRA Install Steps (Working)

This guide reflects the final working command sequence from your setup (`verlai/verl:vllm012.dev4`) and includes the fixes for:
- `ImportError: cannot import name 'VLMLoRA'`
- `ModuleNotFoundError: No module named 'modelopt'`
- `ModuleNotFoundError: No module named 'pulp'`

---

## 1) Start container

```bash
docker run --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v "$(pwd)/verl:/workspace/verl/verl" \
  --name verl3 -it --entrypoint /bin/bash \
  verlai/verl:vllm012.dev4
```

---

## 2) Create and activate venv (with system packages visible)

```bash
cd /workspace/verl/verl
apt-get update && apt-get install -y python3-venv python3-pip

python3.10 -m venv --system-site-packages /opt/verl-venv
source /opt/verl-venv/bin/activate

python3 -m pip install -U pip
```

---

## 3) Install local veRL package (editable, no dependency overwrite)

```bash
cd /workspace/verl/verl
python3 -m pip install --no-deps -e .
```

---

## 4) Install Megatron-Bridge compatible with current veRL (VLMLoRA present)

```bash
python -m pip uninstall -y megatron-bridge mbridge megatron_bridge || true
python -m pip install --no-deps --force-reinstall \
  "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@83a7c1134c562d8c6decd10a1f0a6e6a7a8a3a44"
```

---

## 5) Install missing Bridge/ModelOpt runtime deps discovered at import-time

```bash
python -m pip install --no-deps "nvidia-modelopt==0.41.0"
python -m pip install "PuLP==3.3.0"
```

---

## 6) Verify the stack

```bash
python - <<'PY'
import modelopt.torch.distill as mtd
import megatron.bridge.peft.lora as l
print("VLMLoRA present:", hasattr(l, "VLMLoRA"))
from megatron.bridge.peft.lora import LoRA, VLMLoRA
print("imports OK")
PY
```

Expected success signal:
- `VLMLoRA present: True`
- `imports OK`

Possible warning you can ignore for this workflow:
- `Failed to import vllm plugin ... vllm.attention has no attribute 'Attention'` (if not using that ModelOpt plugin path).

---

## 7) Restart Ray processes before rerunning training

```bash
ray stop -f || true
pkill -f "ray::" || true
bash train.sh
```

---

## Notes

- Keep using `--no-deps` for local `verl` install and for Megatron-Bridge pinning to avoid clobbering the container’s curated dependency set.
- If you later change Megatron-Bridge version, rerun the import verification block before launching training.

1.5B works: run_qwen2-1.5b_math_megatron_lora
Try 3B: 
