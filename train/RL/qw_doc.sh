
workdir=/media/cheliu/cl_code/Entropy
export filter=False
export rewardtype=custom
export lossmode=vanilla
export nsample=8
export basemodelname=math
export ppo_micro=2
export maxlen=2500
export maxtoken=32768
export ase=False
export sse=False
export coeff=0.0
export temp=1.0
export entropy=none
export explore=False
export explore_loss=none
export rbsz=64
export tbsz=32
export llama_ins_zero=False
export zero=False 
export judge=False
export doc=True 
export adv=grpo
export lr=5e-7
export cliphigh=0.2
# export name=qwbase
# export maxpixels=1605632
# export maxpixels=1204224
export maxpixels=548800
export maxin=13000
export limit_images=20
export MODEL_PATH=/media/cheliu/cl_code/Entropy/share_base_models/stage1_11k_2epoch 

export trainfile=/media/cheliu/cl_code/Entropy/doc_related_data/stage1_retrieval_data/train/stage1_rl_new_changed.parquet
export valfile="/media/cheliu/cl_code/Entropy/doc_related_data/stage1_retrieval_data/test/stage1_rl_filtered_sampled_changed.parquet"

export tagname=docrlretrieval_sftRL
export sft_template=retrieval
bash ${workdir}/recipe/dapo/vl_baidu.sh

