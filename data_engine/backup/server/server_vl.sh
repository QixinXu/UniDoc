set -ex

echo "--- Debugging Environment ---"
echo "User: $(whoami)"
echo "Conda Env: $CONDA_PREFIX"
echo "NVCC Path: $(which nvcc)"
echo "NVCC Version:"
nvcc --version
echo "--> LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "---------------------------"

export HOST_IP=0.0.0.0
port=2333
model="PATH/TO/Qwen2.5-VL-72B-Instruct"

vllm serve $model --max-model-len 16384 --port $port -tp 4 \
  --trust-remote-code --gpu-memory-utilization 0.6 --limit-mm-per-prompt "image=20,video=0" \
  --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 401408}'