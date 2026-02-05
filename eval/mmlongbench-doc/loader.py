import json
import pandas as pd
import uuid
from datasets import load_dataset
from tqdm import tqdm
import ast

with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/mmlongbench-doc/concatenated_images_data.json', 'r') as f:
    data = json.load(f)

results = []
for item in tqdm(data):
    question = item['question']
    answer = item['answer']
    evidence_pages = ast.literal_eval(item['evidence_pages'])
    images = item['concatenated_images']
    
    final_question = "<image>" * len(images) + question + "\nLet's think step by step and output the final answer within \\boxed{}."
    
    result = dict(
        data_source='mmlongbench-doc',
        images=images,
        prompt=[dict(role='user', content=final_question)],
        ability='documents_reasoning',
        reward_model={
            'ground_truth': answer,
            'style': 'rule'
        },
        extra_info={'index': str(uuid.uuid4()), 'evidence_pages': evidence_pages}
    )
    
    results.append(result)

df = pd.DataFrame(results)
df.to_parquet("/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/mmlongbench-doc/mmlongbench-doc_test.parquet")

# print the first item in the dataset
ds = load_dataset('parquet', data_files='/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/mmlongbench-doc/mmlongbench-doc_test.parquet')
print(ds['train'][0])
    