import json

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/dude_val.json') as f:
    data = json.load(f)

new_data = []
for item in data:
    questionId = item['qid']
    question = item['question']
    assert type(item['answer']) == str
    answers = [item['answer']]
    answer_type = item['answer_format']
    doc_id = item['doc_id']
    data_split = "val"
    new_data.append({
        "questionId": questionId,
        "question": question,
        "answers": answers,
        "answer_type": answer_type,
        "doc_id": doc_id,
        "data_split": data_split
    })
    
full_data = {
    "dataset_name": "DUDE Dataset",
    "dataset_version": "0.1",
    "data": new_data
}

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/DUDEeval/gt/val_gt_full.json', 'w') as f:
    json.dump(full_data, f, indent=4, ensure_ascii=False)