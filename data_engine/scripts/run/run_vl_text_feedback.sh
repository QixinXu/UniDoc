set -ex
export HOST_IP=0.0.0.0
port=2331
rank=0
total=1
cname=vl_config_text_feedback.yaml

python synthesize_from_feedback.py --port $port --rank ${rank} --total-ranks ${total} --config_file $cname # port rank total
