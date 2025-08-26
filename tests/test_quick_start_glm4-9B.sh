#!/bin/bash

cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .

hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

cd /root/slime
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist

unset http_proxy
unset https_proxy

cd /root/slime
bash scripts/run-glm4-9B.sh
