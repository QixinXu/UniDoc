import jsonlines
import json
import os

# configuration
input_file = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/results/dude/syndata_ablv2_dude_real_extracted.json'

file_id = input_file.split('/')[-1].split('.')[0]
output_dir = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/submissions'
gt_output_dir = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/gt'

with open(input_file) as f:
    data = json.load(f)

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/gt/val_gt_full.json') as f:
    gt_data = json.load(f)['data']

unans_lst = [
    'No information',
    'No specific',
    'Cannot determine',
    'Not specified',
    "Not mentioned",
    "Not provided",
]
results = []
gt_results = []
for item in data:
    # question = item['prompt'][0]['content'].replace('<image>', '')
    # for gt_item in gt_data:
    #     if question == gt_item['question']:
    #         results.append({
    #             'questionId': gt_item['questionId'],
    #             'answers': [item['prediction'][0]]
    #         })
    #         gt_results.append(gt_item)
    #         break
    question = item['prompt'][0]['content'].replace('<image>', '')
    for gt_item in gt_data:
        if question == gt_item['question']:
            short_answer = item['short_answer']
            try:
                short_answer = json.loads(short_answer)['short_answer']
            except:
                short_answer = short_answer.split('"short_answer"')[-1].split('\n')[0].replace('": "', '').strip('"')
            
            if short_answer == 'Not Answerable':
                short_answer = ""
            else:
                for unans_str in unans_lst:
                    if unans_str.lower() in short_answer.lower():
                        short_answer = ""
                if short_answer.startswith('No ') or short_answer.startswith('Not '):
                    short_answer = ""
                if short_answer == "None":
                    short_answer = ""
            results.append({
                'questionId': gt_item['questionId'],
                'answers': [short_answer]
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

print(f"python /map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/evaluate_submission.py -s {output_file} -g {gt_output_file}")
