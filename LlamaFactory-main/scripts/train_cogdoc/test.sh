#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_yaml_config>"
  echo "Example: $0 /map-vepfs/haozhe/xuqixin/LLaMA-Factory/examples/train_full/stage2.yaml"
  exit 1
fi

YAML_PATH="$1"

export SWANLAB_API_KEY="W81ICgy0oekxdxH8SAotg"

cd /map-vepfs/haozhe/xuqixin/LLaMA-Factory

source /map-vepfs/miniconda3/bin/activate /map-vepfs/miniconda3/envs/doc_xqx

llamafactory-cli train "$YAML_PATH"