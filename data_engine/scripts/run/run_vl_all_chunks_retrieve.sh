#!/bin/bash
set -ex

# Set default values
CHUNKS_DIR=""
BASE_OUTPUT_PREFIX=""
TEMPLATE_CONFIG="/map-vepfs/haozhe/xuqixin/thinking-with-documents/scripts/configs/vl_config_retrieve_thinking.yaml"
port=2333
rank=0
total=1

# Parse command-line options
while getopts "c:g:" opt; do
  case $opt in
    c)
      CHUNKS_DIR="$OPTARG"
      ;;
    g)
      BASE_OUTPUT_PREFIX="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$CHUNKS_DIR" ] || [ -z "$BASE_OUTPUT_PREFIX" ]; then
  echo "Usage: $0 -c <chunks_dir> -g <output_prefix>"
  exit 1
fi

# Ensure output directory exists
mkdir -p "$BASE_OUTPUT_PREFIX"

# Loop through all JSON files in the specified directory
for chunk_file in "$CHUNKS_DIR"/*.json; do
  [ -e "$chunk_file" ] || continue

  echo "================================================================"
  echo ">>> [INFO] Processing file: $chunk_file"
  echo "================================================================"

  filename=$(basename -- "$chunk_file")
  filename_no_ext="${filename%.*}"

  output_prefix="${BASE_OUTPUT_PREFIX}${filename_no_ext}_"

  temp_config_file="text_feedback_retrieve_temp_config_${filename_no_ext}.yaml"

  sed -e "s|__INPUT_FILE_PATH__|${chunk_file}|g" \
      -e "s|__OUTPUT_FILE_PREFIX__|${output_prefix}|g" \
      "$TEMPLATE_CONFIG" > "$temp_config_file"

  echo ">>> [INFO] Generated temporary config file: $temp_config_file"
  echo ">>> [INFO] Input path: ${chunk_file}"
  echo ">>> [INFO] Output prefix: ${output_prefix}"

  python /map-vepfs/haozhe/xuqixin/thinking-with-documents/scripts/synthesize/synthesize_retrieval_thinking.py --port "$port" --rank "$rank" --total-ranks "$total" --config_file "$temp_config_file"

  echo ">>> [SUCCESS] Finished processing $chunk_file"
  echo ""
  
  rm -f "$temp_config_file"
  sleep 0.5  # Optional: sleep for a second to avoid overwhelming the server
done

# Clean up temporary config files
rm -f text_feedback_retrieve_temp_config_*.yaml
echo "================================================================"
echo ">>> [ALL DONE] All chunk files have been processed."
echo "================================================================"