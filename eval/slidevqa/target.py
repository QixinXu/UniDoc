import os
import json
from tqdm import tqdm

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/SlideVQA_train.json') as f:
    data = json.load(f)

new_data = []
for item in tqdm(data):
    images = item['image']
    evidence_pages = item['evidence_pages']
    try:
        target_images = [images[i - 1] for i in evidence_pages]
    except:
        print(f"Error processing item with qid {item['qid']}")
        continue
    for image_path in target_images:
        assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
    qid = item['qid']
    question = item['question']
    answer = item['answer']
    new_item = {
        'qid': qid,
        'question': question,
        'answer': answer,
        'image': target_images
    }
    new_data.append(new_item)

print(len(new_data))
# split the new_data into many json files, each file contains 100 items
os.makedirs('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/synthesize/data/slidevqa_initial_thinking', exist_ok=True)
file_id = 0
for i in range(0, len(new_data), 100):
    chunk = new_data[i:i+100]
    with open(f'/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/synthesize/data/slidevqa_initial_thinking/SlideVQA_train_{file_id:03d}.json', 'w') as f:
        json.dump(chunk, f, indent=4)
    file_id += 1