from datasets import load_dataset
import os
import uuid
import pandas as pd
from tqdm import tqdm
import base64

ds = load_dataset('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA', split="test", cache_dir='/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/cache')
ds = ds.shuffle(seed=42)
total_num = len(ds)
selected_ds = ds.select(range(int(total_num * 0.25)))
        
print(ds[0])

# df = pd.read_parquet('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/data/train-00000-of-00053.parquet')
# data = df.to_dict(orient='records')
# binary_data = data[0]['page_1']['bytes']
# base64_encoded_data = base64.b64encode(binary_data)
# base64_string = base64_encoded_data.decode('utf-8')
# print(base64_string)
# image_root_dir = '/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/SlideVQA/images'
# os.makedirs(image_root_dir, exist_ok=True)
# results = []
# for item in tqdm(selected_ds):
#     question = item['question']
#     answer = item['answer']
#     evidence_pages = item['evidence_pages']
#     images = []
#     deck_name = item['deck_name']
#     image_dir = os.path.join(image_root_dir, deck_name)
#     os.makedirs(image_dir, exist_ok=True)
#     for i in range(1, 20):
#         key = f"page_{i}"
#         assert key in item, f"Key {key} not found in item"
#         image_pil = item[key]
#         image_save_path = os.path.join(image_dir, f"{key}.jpg")
#         if not os.path.exists(image_save_path):
#             image_pil.save(image_save_path)
#         images.append(image_save_path)
#     final_question = "<image>" * len(images) + question + "\nLet's think step by step and output the final answer within \\boxed{}."

#     result = dict(
#         data_source='SlideVQA',
#         images=images,
#         prompt=[dict(role='user', content=final_question)],
#         ability='documents_reasoning',
#         reward_model={
#             'ground_truth': answer,
#             'sytle': 'rule'
#         },
#         extra_info={'index': str(uuid.uuid4()), 'evidence_pages': evidence_pages} 
#     )
    
#     results.append(result)
    
# df = pd.DataFrame(results)
# df.to_parquet("/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/SlideVQA/SlideVQA_test.parquet")

# # print the first item in the dataset
# ds = load_dataset('parquet', data_files='/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/SlideVQA/SlideVQA_test.parquet')
# print(ds['train'][0])
    