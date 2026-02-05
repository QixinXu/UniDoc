set -x

cd /map-vepfs/haozhe/xuqixin/thinking-with-documents

source /map-vepfs/miniconda3/bin/activate /map-vepfs/miniconda3/envs/doc_xqx

bash scripts/server/server_vl_blackbox.sh

sleep 150

bash scripts/run/run_vl_all_chunks_retrieve.sh \
    -c data/synthesize/data/stage1_retrival/data \
    -g data/synthesize/data/stage1_retrival/syn_retrieve_thinking_chunks/
