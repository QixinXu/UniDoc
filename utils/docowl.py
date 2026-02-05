import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
from tqdm import tqdm
import json
import random

class DocOwlInfer():
    def __init__(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')
        
    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer


docowl = DocOwlInfer(ckpt_path='/map-vepfs/haozhe/xuqixin/thinking-with-documents/models/DocOwl2')

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/SlideVQA_test.json') as f:
    data = json.load(f)

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/results/slidevqa/docowl.json') as f:
    existing_data = json.load(f)

existing_qids = [item['qid'] for item in existing_data]
non_existing_data = [item for item in data if item['qid'] not in existing_qids]

data = random.sample(non_existing_data, 150)

# open an output file
with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/results/slidevqa/docowl1.json', 'w') as f_out:

    for item in tqdm(data):
        images = item['image']
        query = item['question']
        answer = docowl.inference(images, query=query)
        item['pred'] = answer
        f_out.write(json.dumps(item) + '\n')
        f_out.flush()
        time.sleep(1.0)

# with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/results/slidevqa/docowl.json', 'w') as f:
#     json.dump(new_data, f, indent=4)
    
