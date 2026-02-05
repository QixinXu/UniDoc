set -ex
export HOST_IP=0.0.0.0
port=2331
model="/home/ma-user/work/haozhe/workspace/Entropy-Mechanism-of-RL/LLaMA-Factory/checkpoints/qwen2_5vl-7b/mmlongbench_train_syn_query_gpt_polished_qwen72b_text_feedback_qwen25_sft/checkpoint-660"

vllm serve $model --max-model-len 16384 --port $port -tp 4 \
  --trust-remote-code --gpu-memory-utilization 0.6 --limit-mm-per-prompt "image=20,video=0" \
  --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 401408}'
# python -m vllm.entrypoints.openai.api_server --enable-image-input --model $model --port $port --max-model-len 20000 -tp 1 --gpu-memory-utilization 0.6 --trust-remote-code & # --disable-log-stats