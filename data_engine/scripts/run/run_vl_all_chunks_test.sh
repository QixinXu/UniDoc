#!/bin/bash
set -ex

CHUNKS_DIR="/home/ma-user/work/haozhe/workspace/o1_reflection/release/chunks_test"

BASE_OUTPUT_PREFIX="/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/results/test/mmlongbench_train_syn_query_gpt_polished_qwen72b_text_feedback_qwen25_vl_sft/"

TEMPLATE_CONFIG="/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/configs/vl_config_template_test.yaml"

port=2331
rank=0
total=1


mkdir -p $BASE_OUTPUT_PREFIX

for chunk_file in "$CHUNKS_DIR"/*.parquet; do
  [ -e "$chunk_file" ] || continue

  echo "================================================================"
  echo ">>> [INFO] Processing file: $chunk_file"
  echo "================================================================"

  filename=$(basename -- "$chunk_file")
  filename_no_ext="${filename%.*}"

  output_prefix="${BASE_OUTPUT_PREFIX}${filename_no_ext}_"

  temp_config_file="test_temp_config_${filename_no_ext}.yaml"

  sed -e "s|__INPUT_FILE_PATH__|${chunk_file}|g" \
      -e "s|__OUTPUT_FILE_PREFIX__|${output_prefix}|g" \
      "$TEMPLATE_CONFIG" > "$temp_config_file"

  echo ">>> [INFO] Generated temporary config file: $temp_config_file"
  echo ">>> [INFO] Input path: ${chunk_file}"
  echo ">>> [INFO] Output prefix: ${output_prefix}"

  python /home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/scripts/synthesize/synthesize_vl_test.py --port $port --rank ${rank} --total-ranks ${total} --config_file "$temp_config_file"

  echo ">>> [SUCCESS] Finished processing $chunk_file"
  echo ""
  
  rm -f "$temp_config_file"
  sleep 0.5  # Optional: sleep for a second to avoid overwhelming the server
done

# Clean up temporary config files
rm -f test_temp_config_*.yaml
echo "================================================================"
echo ">>> [ALL DONE] All chunk files have been processed."
echo "================================================================"