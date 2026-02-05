import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from qwen_vl_utils import process_vision_info 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MAX_BATCH_SIZE = 10 
MIN_PIXEL = 512*28*28
MAX_PIXEL = 4096*28*28

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/map-vepfs/models/Qwen2.5-VL-7B-Instruct/",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("/map-vepfs/models/Qwen2.5-VL-7B-Instruct/", min_pixels=MIN_PIXEL, max_pixels=MAX_PIXEL)
processor.tokenizer.padding_side = 'left'

select_id = processor.tokenizer.convert_tokens_to_ids("A")
def post_proc_func(x):
    logits = x['scores'][0]
    scores = F.softmax(logits, dim=-1)[:, select_id].detach()
    return scores

with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/stage1_testset.jsonl') as f:
    data = [json.loads(line) for line in f]

exact_match = 0
recall_all = 0
recall_miss = 0
    
for item in tqdm(data, desc="Processing data"):
    images = item["stage1_images"]
    target_images = item["target_images"]
    question = item["question"]
    
    question_prompt = f"Question: {question.strip()}\n"
    task_prompt = "Does the information within the image provide the necessary details to accurately answer the given question?\n"
    post_prompt = f"A. yes\nB. no\n"
    final_prompt = question_prompt + task_prompt + post_prompt
    
    all_scores = []
    for i in tqdm(range(0, len(images), MAX_BATCH_SIZE), desc="Processing chunks", leave=False):
        chunk_images = images[i:i + MAX_BATCH_SIZE]
        
        batch_messages = []
        for image_path in chunk_images:
            message_for_one_image = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": final_prompt}]}]
            batch_messages.append(message_for_one_image)

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)

        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generation_output = model.generate(
                **inputs, max_new_tokens=32, output_scores=True, return_dict_in_generate=True
            )

        chunk_scores = post_proc_func(generation_output).cpu()
        all_scores.append(chunk_scores)

        del inputs, generation_output, chunk_scores
        torch.cuda.empty_cache()

    scores = torch.cat(all_scores)
    print(scores)
    
    k = min(2, len(images))
    topk_scores, topk_indices = torch.topk(scores, k=k)
    selected_images = [images[idx] for idx in topk_indices.tolist()]
    
    if set(target_images).issubset(set(selected_images)):
        if set(target_images) == set(selected_images):
            exact_match += 1
        recall_all += 1
    else:
        recall_miss += 1

len_data = len(data)
print(F"max pixels: {MAX_PIXEL}")
print(F"min pixels: {MIN_PIXEL}")
print(f"Total samples: {len_data}")
print(f"Exact match: {exact_match}, {exact_match/len_data:.2%}")
print(f"Recall all: {recall_all}, {recall_all/len_data:.2%}")
print(f"Recall miss: {recall_miss}, {recall_miss/len_data:.2%}")
