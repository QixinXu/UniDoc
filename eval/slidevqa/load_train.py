from datasets import load_dataset
import os
from tqdm import tqdm

ds = load_dataset('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA', split="test", cache_dir='/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/cache')

# # print(len(ds))
# print(ds[0])
# breakpoint()

results = []
for item in tqdm(ds):
    qid = 'slidevqatest_' + f'{item["deck_name"]}' + f'_{item["qa_id"]}'
    question = item['question']
    answer = item['answer']
    image = []
    image_save_dir = f"/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/test_images/{item['deck_name']}"
    os.makedirs(image_save_dir, exist_ok=True)
    for i in range(1, 21):
        key = f"page_{i}"
        if key not in item:
            continue
        image_pil = item[key]
        image_save_path = f"{image_save_dir}/{key}.jpg"
        if not os.path.exists(image_save_path):
            image_pil.save(image_save_path)
        image.append(image_save_path)
    pdf_path = item['deck_url']
    evidence_pages = item['evidence_pages']
    result = {
        'qid': qid,
        'question': question,
        'answer': answer,
        'image': image,
        'pdf_path': pdf_path,
        'evidence_pages': evidence_pages
    }
    results.append(result)

with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/SlideVQA/SlideVQA_test.json', 'w') as f:
    import json
    json.dump(results, f, indent=4)
    
