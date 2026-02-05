import io
import os
import re
import ast
import sys
import copy
import json
import time
import math
import torch
import base64
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, UnidentifiedImageError
random.seed(42)

def encode_image_to_base64(image_path):
    """
    Encodes a local image file into base64, with a prior integrity check.
    """
    if not os.path.exists(image_path):
        print(f"⚠️ Error: Image file not found at {image_path}", file=sys.stderr)
        return None

    try:
        # Step 1: Read the image file content into memory.
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Step 2: Use an in-memory stream to check the image integrity with Pillow.
        # This simulates how 'qwen_vl_utils' would process the data.
        with io.BytesIO(image_bytes) as bio:
            Image.open(bio).verify()  # The .verify() method checks the file integrity without fully decoding it.
        
        # If the integrity check passes, proceed with Base64 encoding.
        return base64.b64encode(image_bytes).decode('utf-8')

    except UnidentifiedImageError:
        print(f"⚠️ Error: Corrupted or unidentified image file at {image_path}", file=sys.stderr)
        return None
    
    except Exception as e:
        print(f"⚠️ Error encoding image {image_path}: {e}", file=sys.stderr)
        return None

def main(args):
    """
    Main function to run the vision-language model benchmark.
    """
    start_time = time.time()
    print(f"Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 30, "video": 0},
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8,
        max_num_seqs=64
    )
    print(f"Max model length: {llm.llm_engine.model_config.max_model_len}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        repetition_penalty=1.05,
        max_tokens=2048,
        stop_token_ids=[],
    )

    # --- Data Loading and Preparation ---
    print(f"Loading '{args.benchmark}' benchmark data...")
    if args.benchmark == 'slidevqa':
        ds = load_dataset('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA', split="test", cache_dir='/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/SlideVQA/cache')
        ds = ds.shuffle(seed=42)
        total_num = len(ds)
        selected_ds = ds.select(range(int(total_num * 0.25)))
        all_data = ds.to_list()
        if args.sample_rate < 1.0:
            random.shuffle(all_data)
            all_data = all_data[:int(len(all_data) * args.sample_rate)]
        name_key = 'deck_name'
    elif args.benchmark == 'vidoseek':
        name_key = 'uid'
        with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/ViDoSeek/vidoseek_images.json', 'r') as f:
            data = json.load(f)
            all_data = data['examples']
        with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/ViDoSeek/skip_uids.txt', 'r') as f:
            skip_uids = set(line.strip() for line in f)
        all_data = [item for item in all_data if item[name_key] not in skip_uids]
        random.shuffle(all_data)
        all_data = all_data[:int(len(all_data) * 0.25)]
    elif args.benchmark == 'mmlongbench-doc':
        # with open('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/mmlongbench-doc/concatenated_images_data.json', 'r') as f:
        #     all_data = json.load(f)
        # name_key = 'question'  # 'question' is the unique identifier for mmlongbench-doc
        # random.shuffle(all_data)  # Shuffle the dataset for randomness
        # all_data = all_data[:int(len(all_data) * 0.25)]
        if not args.stage1:
            with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/concatenated_images_data_with_target.json', 'r') as f:
                all_data = json.load(f)
        else:
            if not args.unanswerable:
                with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/stage1_testset.jsonl', 'r') as f:
                    all_data = [json.loads(line) for line in f.readlines()]
            else:
                # use the unanswerable version of the dataset
                with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/stage1_testset_unanswerable.jsonl', 'r') as f:
                    all_data = [json.loads(line) for line in f.readlines()]
        if args.sample_rate < 1.0:
            random.shuffle(all_data)
            all_data = all_data[:int(len(all_data) * args.sample_rate)]
        name_key = 'question'
    elif args.benchmark == 'mp-docvqa':
        ds = load_dataset('/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/MP-DocVQA', split='val', cache_dir='/home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/data/MP-DocVQA/cache')
        ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness
        total_num = len(ds)
        selected_ds = ds.select(range(int(total_num * 0.25)))  # sample ratio: 0.25
        all_data = selected_ds.to_list() 
        name_key = 'questionId'  
    elif args.benchmark == 'dude':
        if not args.unanswerable:
            with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/dude_val_with_target.json') as f:
                all_data = json.load(f)
                # all_data = [item for item in all_data if item['doc_type'] != 'not-answerable' or item['evidence_pages'] != '[]']
        else:
            with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/DUDE/dude_val_unanswerable.json') as f:
                all_data = json.load(f)
        if args.sample_rate < 1.0:
            random.shuffle(all_data)
            all_data = all_data[:int(len(all_data) * args.sample_rate)]
        name_key = 'qid'
    elif args.benchmark == 'synthesize_figure':
        with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/synthesized_queries_figure_filter.json', 'r') as f:
            all_data = json.load(f)
        random.shuffle(all_data)
        all_data = all_data[:int(len(all_data) * 0.25)]
        name_key = 'qid'
    elif args.benchmark == 'synthesize_table':
        with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/synthesized_queries_table.json', 'r') as f:
            all_data = json.load(f)
        random.shuffle(all_data)
        name_key = 'qid'
    elif args.benchmark == 'stage1_val_all':
        ds = load_dataset("parquet", data_files='/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/allstage_val/stage1_rl_filtered_sampled_changed.parquet', cache_dir="/map-vepfs/haozhe/data_conversion/datasets_cache")['train']
        all_data = ds.to_list()
        name_key = None
    elif args.benchmark == 'stage2_val_all':
        ds = load_dataset("parquet", data_files='/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/allstage_val/s2_rl_valset.parquet', cache_dir="/map-vepfs/haozhe/data_conversion/datasets_cache")['train']
        all_data = ds.to_list()
        name_key = None
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    
    all_data *= args.repeat_times  # Repeat each entry if specified

    # --- Resume from Previous Run ---
    if os.path.exists(args.output_path):
        try:
            old = [json.loads(line).get(name_key) for line in open(args.output_path).readlines()]
            seen = set(filter(None, old))
            data = [x for x in all_data if x.get(name_key) not in seen]
            print(f"Found {len(seen)} already processed entries in {args.output_path}. Resuming with {len(data)} new entries.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Could not read progress from {args.output_path}. Error: {e}")
            data = all_data
    else:
        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)
        data = all_data
        print(f"No existing output file found at {args.output_path}. Starting from scratch with {len(data)} entries.")

    if not data:
        print("No new data to process. Exiting.")
        return

    # --- Prepare Inputs for LLM ---
    llm_inputs_lst = []
    for entry in tqdm(data, desc="Preparing inputs"):
        if args.benchmark == 'slidevqa':
            query = entry['question']
            assert not (args.cot and args.new_cot), "Only one of --cot and --new_cot can be set."
            if args.cot:
                query += cot_instruction
            if args.new_cot:
                query = new_cot_instruction.format(question=query)
            if args.stage1:
                assert not args.cot and not args.new_cot, "For stage1 retrieval, --cot and --new_cot must be False."
                if args.stage1_new_prompt:
                    query = new_retrieval_prompt.format(question=query)
                else:
                    query = retrieval_prompt.format(question=query)
            query += unanswerable_prompt if args.unanswerable_prompt else ''
            
            content_list = [{"type": "text", "text": query}]
            pages_lst = range(1, 21) if not args.target_images else entry['evidence_pages']
            for index in pages_lst:
                key = f"page_{index}"
                assert key in entry, f"Expected key '{key}' not found in entry."
                bytes_ = entry[key]['bytes']
                base64_image = base64.b64encode(bytes_).decode('utf-8')
                if base64_image:
                    if not args.stage1:
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                        #     "min_pixels": 224 * 224,
                        #     "max_pixels": 1024 * 28 * 28
                        })
                    else:
                        content_list.append({
                            "type": "text",
                            "text": f"[Image {index}]: "
                        })
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                            "min_pixels": 224 * 224,
                            "max_pixels": 1024 * 28 * 28
                        })
            entry_key = 'deck_name'
        elif args.benchmark == 'vidoseek':
            assert not (args.cot and args.new_cot), "Only one of --cot and --new_cot can be set."
            if args.cot:
                query += cot_instruction
            if args.new_cot:
                query = new_cot_instruction.format(question=query)
            query += unanswerable_prompt if args.unanswerable_prompt else ''
            content_list = [{"type": "text", "text": query}]
            images = entry['images']
            for image_path in images:
                content_list.append({
                    "type": "image",
                    "image": f"file://{image_path}"
                })
            entry_key = 'uid'
        elif args.benchmark == 'mmlongbench-doc':
            query = entry['question']
            # query = 'Read the above documents and answer this question:\n' + query + '\nPlease make your answer as concise as possible.'
            assert not (args.cot and args.new_cot), "Only one of --cot and --new_cot can be set."
            if args.cot:
                query += cot_instruction
            if args.new_cot:
                query = new_cot_instruction.format(question=query)
            if args.stage1:
                assert not args.cot and not args.new_cot, "For stage1 retrieval, --cot and --new_cot must be False."
                if args.stage1_new_prompt:
                    query = new_retrieval_prompt.format(question=query)
                else:
                    query = retrieval_prompt.format(question=query)
            
            content_list = [{"type": "text", "text": query}]
            if not args.stage1:
                images = entry['concatenated_images'].copy() if not args.target_images else entry['target_images'].copy()
            else:
                images = entry['stage1_images'].copy()
            
            
            if args.add_noise_target:
                assert args.target_images, "To add noise to target images, --target_images must be set to True."
                assert args.noise_count > 0, "To add noise to target images, --noise_count must be greater than 0."
                full_images = copy.deepcopy(entry['images'])
                for image_ in entry['target_images']:
                    if image_ in full_images:
                        full_images.remove(image_)
                noisy_images = random.sample(full_images, min(args.noise_count, len(full_images), 25 - len(images)))
                images.extend(noisy_images)
                if args.shuffle:
                    random.shuffle(images)
                
            if args.replace_target:
                assert args.target_images, "To replace target images, --target_images must be set to True."
                assert 0.0 < args.replace_ratio <= 1.0, "To replace target images, --replace_ratio must be in (0.0, 1.0]."
                assert args.replace_base_count >= 0, "To replace target images, --replace_base_count must be >= 0."
                full_images = copy.deepcopy(entry['images'])
                for image_ in entry['target_images']:
                    if image_ in full_images:
                        full_images.remove(image_)
                replace_num = len(images) * args.replace_ratio
                fractional_part, integer_part = math.modf(replace_num)
                random_ = random.random()
                actual_replace_num = int(integer_part) + 1 if random_ < fractional_part else int(integer_part)
                actual_remain_num = len(images) - actual_replace_num
                remain_images = random.sample(images, actual_remain_num) if actual_remain_num > 0 else []
                try:
                    noisy_images = random.sample(full_images, min(len(images) - actual_remain_num + args.replace_base_count, len(full_images))) if (len(images) - actual_remain_num + args.replace_base_count) > 0 else []
                except:
                    print(f"images: {images}")
                    print(f"full_images: {full_images}")
                    print(f"len(images): {len(images)}")
                    print(f"len(full_images): {len(full_images)}")
                    print(f"actual_replace_num: {actual_replace_num}")
                    print(f"actual_remain_num: {actual_remain_num}")
                images = remain_images + noisy_images
                
            for idx, image_path in enumerate(images):
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    if not args.stage1:
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                            # "min_pixels": 224 * 224,
                            # "max_pixels": 1024 * 28 * 28,
                        })
                    else:
                        content_list.append({
                            "type": "text",
                            "text": f"[Image {idx + 1}]: "
                        })
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                            "min_pixels": 224 * 224,
                            "max_pixels": 1024 * 28 * 28,
                        })
            entry_key = 'question'
        elif args.benchmark == 'mp-docvqa':
            query = entry['question'] + cot_instruction if args.cot else entry['question']
            query += unanswerable_prompt if args.unanswerable_prompt else ''
            content_list = [{"type": "text", "text": query}]
            for index in range(1, 21):
                key = f'image_{index}'
                assert key in entry, f"Expected key '{key}' not found in entry."
                if not entry[key]:
                    break
                bytes_ = entry[key]['bytes']
                base64_image = base64.b64encode(bytes_).decode('utf-8')
                if base64_image:
                    content_list.append({
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}"
                    })
            entry_key = 'questionId'
        elif args.benchmark == 'dude':
            query = entry['question']
            assert not (args.cot and args.new_cot), "Only one of --cot and --new_cot can be set."
            if args.cot:
                query += cot_instruction
            if args.new_cot:
                query = new_cot_instruction.format(question=query)
            if args.stage1:
                assert not args.cot and not args.new_cot, "For stage1 retrieval, --cot and --new_cot must be False."
                if args.stage1_new_prompt:
                    query = new_retrieval_prompt.format(question=query)
                else:
                    query = retrieval_prompt.format(question=query)
            query += unanswerable_prompt if args.unanswerable_prompt else ''
            content_list = [{"type": "text", "text": query}]
            images_lst = entry['target_images'] if args.target_images else entry['image']
            if args.stage1:
                images_lst = entry['image']
            for idx, image_path in enumerate(images_lst):
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    if not args.stage1:
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                            # "min_pixels": 224 * 224,
                            # "max_pixels": 1024 * 28 * 28,
                        })
                    else:
                        content_list.append({
                            "type": "text",
                            "text": f"[Image {idx + 1}]: "
                        })
                        content_list.append({
                            "type": "image",
                            "image": f"data:image;base64,{base64_image}",
                            "min_pixels": 224 * 224,
                            "max_pixels": 1024 * 28 * 28,
                        })
            entry_key = 'qid'
        elif args.benchmark == 'synthesize_figure' or args.benchmark == 'synthesize_table':
            # temporary patch
            query = entry['question'] + "Please think step by step and give the final answer in the format of '\nAnswer: <your answer>'."
            content_list = [{"type": "text", "text": query}]
            assert args.target_images, "For 'synthesize' benchmark, --target_images must be set to True."
            images = entry['related_pages_path']
            for image_path in images:
                base64_image = encode_image_to_base64(image_path)
                if base64_image:
                    content_list.append({
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                        "min_pixels": 224 * 224,
                        "max_pixels": 1024 * 1024,
                    })
            entry_key = 'qid'
        elif args.benchmark == 'stage1_val_all':
            query = entry['prompt'][0]['content']
            images = entry['images']
            image_cnt = query.count('<image>')
            if args.old_stage1_prompt:
                new_query = query.split('[image_1]')[0].strip('Question')[-1].strip()
                new_query = retrieval_prompt.format(question=new_query)
                content_list = [{"type": "text", "text": new_query}]
                for idx, image_item in enumerate(images):
                    bytes_ = image_item['bytes']
                    base64_image = base64.b64encode(bytes_).decode('utf-8')
                    content_list.append({
                        "type": "text",
                        "text": f"[Image {idx + 1}]: "
                    })
                    content_list.append({
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                        # "min_pixels": 224 * 224,
                        "max_pixels": 262144,
                    })
            else:
                new_query = query.split('[image_1]')[0].split('Question:')[-1].strip()
                new_query = new_retrieval_prompt.format(question=new_query)
                content_list = [{"type": "text", "text": new_query}]
                for idx, image_item in enumerate(images):
                    bytes_ = image_item['bytes']
                    base64_image = base64.b64encode(bytes_).decode('utf-8')
                    content_list.append({
                        "type": "text",
                        "text": f"[Image {idx + 1}]: "
                    })
                    content_list.append({
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                        "max_pixels": 262144
                    })

        elif args.benchmark == 'stage2_val_all':
            query = entry['prompt'][0]['content']
            images = entry['images']
            content_list = []
            current_image_idx = 0
            segments = re.split("(<image>)", query)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == '<image>':
                    bytes_ = images[current_image_idx]['bytes']
                    base64_image = base64.b64encode(bytes_).decode('utf-8')
                    content_list.append({
                        "type": "image",
                        "image": f"data:image;base64,{base64_image}",
                        "max_pixels": 1204224
                    })
                    current_image_idx += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            assert current_image_idx == len(images),  f"Number of images does not match! query: {query}, segments: {segments}, content_list: {content_list}"
        else:
            raise ValueError(f"Unsupported benchmark: {args.benchmark}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content_list}
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if args.new_cot:
            prompt += "<think>\n"

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        entry['input'] = prompt
        # filter out overlong inputs
        if args.filter_overlong:
            # torch.set_printoptions(threshold=float('inf'))
            model_inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            # with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/input_len.txt', 'w') as f: 
            #     f.write(f"{prompt}\n")
            #     f.write(f"Length: {len(input_ids[0])}\n") 
            #     f.write(f"Tensor: {input_ids}\n\n")
            # torch.set_printoptions(profile='default')
            entry["input_len"] = len(input_ids[0])
            entry["image_token_count"] = (input_ids[0] == processor.image_token_id).sum().item()
            max_model_len = llm.llm_engine.model_config.max_model_len
            if len(input_ids[0]) > max_model_len:
                print(f"Warning: Entry id: {entry[entry_key]}, Input length {len(input_ids[0])} exceeds max model length ({max_model_len}). Skipping entry.")
                with open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/skip_uids.txt', 'a') as f:  
                    f.write(entry[entry_key] + '\n')
                continue
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        llm_inputs_lst.append(llm_inputs)

    print(f"Generating responses for {len(llm_inputs_lst)} inputs...")
    # generate batch by batch 

    if args.benchmark == 'slidevqa':
        BATCH_SIZE = 64 
        outputs = [] 
        for i in range(0, len(llm_inputs_lst), BATCH_SIZE):
            batch_inputs = llm_inputs_lst[i:i + BATCH_SIZE]
            total_batches = (len(llm_inputs_lst) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Processing batch {i // BATCH_SIZE + 1}/{total_batches} (size {len(batch_inputs)})...")
            batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            outputs.extend(batch_outputs)
    else:
        outputs = llm.generate(llm_inputs_lst, sampling_params=sampling_params)
    print("Generation complete.")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        origin_entry = data[i]
        new_entry = copy.deepcopy(origin_entry)
        new_entry['pred'] = generated_text
        new_entry['model'] = args.model_path
        if 'synthesize' in args.benchmark:
            del new_entry['meta_info']
        if args.benchmark == 'slidevqa':
            for i in range(1, 21):
                key = f"page_{i}"
                if key in new_entry:
                    del new_entry[key]
        if args.benchmark == 'mp-docvqa':
            for i in range(1, 21):
                key = f"image_{i}"
                if key in new_entry:
                    del new_entry[key]
        if args.benchmark == 'dude':
            new_entry['questionId'] = new_entry.pop('qid', None)
            if args.cot:
                generated_text = generated_text.split('<answer>')[-1].strip()
                match = re.search(r'\\boxed{(.*?)}', generated_text)
                if match:
                    final_answer = match.group(1).strip()
                else:
                    final_answer = generated_text.strip()
                new_entry['answers'] = [final_answer]
            else:
                new_entry['answers'] = [generated_text.strip()]
        if args.benchmark == 'stage1_val_all' or args.benchmark == 'stage2_val_all':
            del new_entry['images']
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(outputs)} new results to {args.output_path}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

cot_instruction = "\nLet's think step by step and output the final answer within \\boxed{}."
unanswerable_prompt = '\nIf you think the answer cannot be found in the context, the final answer should be \"Not Answerable\".'
new_cot_instruction = '''
You will be given document pages and a question. You are required to first articulate your reasoning process within <think> tags, and then provide the final answer within <answer> tags.

The thinking process should begin with a thorough overview of the document pages to get the lay of the land. 
Next, analyze the user's question to understand what is being asked.
Proceed to collect evidence from the document. **Crucially, you must perform careful content recognition, you must meticulously read the document and you must ground and list the relevant specific information, data points, details exactly in your thought process. You must not skip this grounding step for recognition** 
After gathering evidence, synthesize and cross-reference the information to formulate your reasoning. 

Your final output should be in the format: <think> Reasoning process here. </think> <answer> Final answer (with \\boxed) here. </answer>

Question: {question}
'''
retrieval_prompt = '''You will be given document pages and a question. Your task is to identify and list all the page numbers of pages that are relevant to answering the question. The chosen page num should be listed within a list, like ["image_xxx", "image_xxx"]. Present your complete thought process within a single and unique `<think></think>` tag and the final answer(a list of page numbers) within a single and unique `<answer></answer>` tag.

Question: {question}
'''
new_retrieval_prompt = '''You will be provided with a question and a set of images, each with a unique identifier. Your task is to analyze the content of each image to determine its relevance to answering the given question. An image is considered relevant if its content (such as graphs, diagrams, figures, or text within the image) provides information, context, or evidence that is necessary or helpful for answering the question. Traverse and analyze each image, then list the identifiers of all relevant images in a list. If no images are relevant, return an empty list []. You must present your complete thought process within a single and unique <think></think> tag, and the final list of relevant image identifiers (e.g., [\"image_A\", \"image_B\"]) within a single and unique <answer></answer> tag.

Question: {question}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vision-Language Model Benchmarks with VLLM.")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the pre-trained model."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default='vidoseek',
        choices=['slidevqa', 'vidoseek', 'mmlongbench-doc', 'mp-docvqa', 'dude', 'synthesize_figure', 'synthesize_table', 'stage1_val_all', 'stage2_val_all'],
        help="The benchmark dataset to use."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSONL file."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="The number of GPUs to use for tensor parallelism."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=1,
        help="Number of times to repeat each input for generation."
    )
    parser.add_argument(
        "--cot",
        action='store_true',
        help="Whether to use Chain-of-Thought (CoT) reasoning in the prompts."
    )
    parser.add_argument(
        "--new_cot",
        action='store_true',
        help="Whether to use the new CoT reasoning prompt."
    )
    parser.add_argument(
        "--filter_overlong",
        action='store_true',
        help="Whether to filter out inputs that exceed the maximum model length."
    )
    parser.add_argument(
        "--target_images",
        action='store_true',
        help="Whether to directly provide target images."
    )
    parser.add_argument(
        "--unanswerable_prompt",
        action='store_true',
        help="Whether to use a specific prompt for unanswerable questions."
    )
    parser.add_argument(
        "--add_noise_target",
        action='store_true',
        help="Whether to add noise to target images."
    )
    parser.add_argument(
        "--noise_count",
        type=int,
        default=2,
        help="Number of noise images to add if --add_noise_target is set."
    )
    parser.add_argument(
        "--replace_target",
        action='store_true',
        help="Whether to replace target images with random images."
    )
    parser.add_argument(
        "--replace_ratio",
        type=float,
        default=0.0,
        help="Number of images to replace if --replace_target is set."
    )
    parser.add_argument(
        "--replace_base_count",
        type=int,
        default=0,
        help="Base number of images to replace if --replace_target is set."
    )
    parser.add_argument(
        "--shuffle",
        action='store_true',
        help="Whether to shuffle the dataset for randomness."
    )
    parser.add_argument(
        "--stage1",
        action='store_true',
        help="Whether to test stage 1 retrieval for each benchmarks."
    )
    parser.add_argument(
        "--stage1_new_prompt",
        default=False,
        action='store_true',
        help="Whether to use the new retrieval prompt for stage 1."
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Sample rate for the dataset."
    )
    parser.add_argument(
        "--unanswerable",
        action='store_true',
        help="Whether to include unanswerable questions in the dataset."
    )
    parser.add_argument(
        "--old_stage1_prompt",
        action='store_true',
        help="Whether to use the old stage 1 prompt."
    )

    args = parser.parse_args()
    main(args)
    
# python inference.py --model_path /home/ma-user/work/haozhe/workspace/o1_reflection/release/base_models/Qwen2.5-VL-7B-Instruct-newest --benchmark slidevqa --output_path /home/ma-user/work/haozhe/xuqixin/code/thinking-with-documents/qwen25vl_slidevqa.jsonl