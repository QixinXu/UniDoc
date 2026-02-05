set -ex
export HOST_IP=0.0.0.0
port=2333
model="/map-vepfs/models/Qwen2.5-VL-72B-Instruct/"

vllm serve $model --max-model-len 16384 --port $port -tp 4 \
  --trust-remote-code --gpu-memory-utilization 0.6 --limit-mm-per-prompt "image=20,video=0" \
  --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 401408}'
