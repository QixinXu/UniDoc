set -ex
export HOST_IP=0.0.0.0
port=2333
rank=0
total=1
cname=data_engine/scripts/configs/vl_config.yaml

python data_engine/scripts/synthesize/synthesize_doc_query_new.py --port $port --rank ${rank} --total-ranks ${total} --config-file $cname --debug --figure --table --pure-text