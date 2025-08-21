1. follow 
docker run  \
  --name sglang_slime_maocheng \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/co-research/shared-models:/root/.cache \
  -it zhuzilin/slime:latest /bin/bash


2. git clone https://github.com/maocheng23/slime.git
cd slime
pip install -e .

3. hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

hf download font-info/qwen3-4b-sft --local-dir /root/font-info/qwen3-4b-sft


4. 
source scripts/models/qwen3-4B-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/font-info/qwen3-4b-sft \
    --save /root/font-info/qwen3-4b-sft_torch_dist


5. bash examples/retool/run_qwen3_4B.sh 