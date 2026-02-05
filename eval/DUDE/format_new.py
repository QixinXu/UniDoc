import jsonlines
import json
import os
import re

# configuration
input_file = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/results/dude/zeroRL_s220_dude_real.json'

file_id = input_file.split('/')[-1].split('.')[0]
output_dir = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/submissions'
gt_output_dir = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/gt'

with open(input_file) as f:
    data = json.load(f)

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/gt/val_gt_full.json') as f:
    gt_data = json.load(f)['data']


def extract_box(answer_str: str):
    # if "No " in answer_str or " no " in answer_str:
    #     return ""
    match = re.search(r'\\boxed\{(.*?)\}', answer_str, re.DOTALL)
    if match:
        short_answer = match.group(1).strip()
        if "No " in short_answer or " no " in short_answer or "not provided" in short_answer or 'not directly' in short_answer or 'The document does not ' in short_answer:
            return ""
        else:
            return short_answer
    return answer_str

results = []
gt_results = []
for item in data:
    question = item['prompt'][0]['content'].replace('<image>', '')
    for gt_item in gt_data:
        if question == gt_item['question']:
            results.append({
                'questionId': gt_item['questionId'],
                'answers': [extract_box(item['stage2_outputs'][0])],
                'gt': gt_item['answers'],
            })
            gt_results.append(gt_item)
            break

gt_final_results = {
    "dataset_name": "DUDE Dataset",
    "dataset_version": "0.1",
    "data": gt_results
}

output_file = os.path.join(output_dir, f'{file_id}.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

gt_output_file = os.path.join(gt_output_dir, f'gt_{file_id}.json')
with open(gt_output_file, 'w') as f:
    json.dump(gt_final_results, f, indent=4, ensure_ascii=False)
    
