#!/bin/bash
export SWANLAB_API_KEY="W81ICgy0oekxdxH8SAotg"

cd /map-vepfs/haozhe/xuqixin/LLaMA-Factory

source /map-vepfs/miniconda3/bin/activate /map-vepfs/miniconda3/envs/doc_xqx

llamafactory-cli train /map-vepfs/haozhe/xuqixin/LLaMA-Factory/examples/train_full/stage2.yaml

