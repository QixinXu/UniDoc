import os
import jsonlines

pdf_paths = []
for jsonl_file in os.listdir('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/synthesize/data/syn_vl_queries_all_0815/raw'):
    if jsonl_file.endswith('.jsonl'):
        file_path = os.path.join('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/synthesize/data/syn_vl_queries_all_0815/raw', jsonl_file)
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                pdf_paths.append(obj['pdf_path'])

pdf_paths = list(set(pdf_paths))  # Remove duplicates
json_paths = [path.replace('.pdf', '.json') for path in pdf_paths]
new_paths = []
with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/arxiv_jsons_with_pdf.txt', 'r') as f:
    for line in f:
        json_path = line.strip()
        if json_path in json_paths:
            print("already exists:", json_path)
        else:
            new_paths.append(json_path)

with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/arxiv_jsons_with_pdf1.txt', 'a') as f:
    for path in new_paths:
        f.write(path + '\n')