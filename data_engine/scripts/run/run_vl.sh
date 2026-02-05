set -ex
export HOST_IP=0.0.0.0
port=2333
rank=0
total=1
cname=/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/scripts/configs/vl_config.yaml

python /home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/scripts/synthesize/synthesize_doc_query_new.py --port $port --rank ${rank} --total-ranks ${total} --config-file $cname --debug --figure --table --pure-text
