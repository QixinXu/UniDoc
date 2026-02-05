import json
import ast

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/testset.jsonl') as f:
    data = [json.loads(line) for line in f]

new_data = []
for item in data:
    evidence_pages = ast.literal_eval(item['evidence_pages'])
    if not evidence_pages:
        continue
    max_page = max(evidence_pages)
    if max_page > 20:
        continue
    stage1_pages = item['images'][:20]
    item['stage1_images'] = stage1_pages
    new_data.append(item)

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/stage1_testset.jsonl', 'w') as f:
    for item in new_data:
        f.write(json.dumps(item) + '\n')