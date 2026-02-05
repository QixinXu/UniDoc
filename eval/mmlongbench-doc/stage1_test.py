import json
import ast
import argparse

def process_jsonl(input_file):

    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    format_error = 0
    exact_match = 0
    recall_all = 0
    recall_miss = 0
    recall_miss_unans = 0
    for item in data:
        target_images = item['target_images']
        stage1_images = item['stage1_images']
        try:
            gt_target_images = []
            for target_image in target_images:
                assert target_image in stage1_images, f"Target image {target_image} not in stage1 images."
                page_rank = stage1_images.index(target_image) + 1
                gt_target_images.append(f'image_{page_rank}')
        except:
            print(item['question'])
            continue
        # print(f"GT target images: {gt_target_images}")
        
        pred = item['pred']
        final_answer = pred.split('<answer>')[-1].split('</answer>')[0].strip()
        try:
            pred_target_images = ast.literal_eval(final_answer)
        except:
            format_error += 1
            continue
        # print(f"Predicted target images: {pred_target_images}")
        try:
            if set(gt_target_images).issubset(set(pred_target_images)):
                if set(pred_target_images) == set(gt_target_images):
                    exact_match += 1
                    recall_all += 1
                else:
                    recall_all += 1
            else:
                recall_miss += 1
                if pred_target_images == []:
                    recall_miss_unans += 1
        except:
            print(item)
            format_error += 1
            continue
        
    len_data = len(data)
    print(f"Total samples: {len_data}")
    print(f"Format error: {format_error}, {format_error/len_data:.2%}")
    print(f"Exact match: {exact_match}, {exact_match/len_data:.2%}")
    print(f"Recall all: {recall_all}, {recall_all/len_data:.2%}")
    print(f"Recall miss: {recall_miss}, {recall_miss/len_data:.2%}")
    print(f"Recall miss due to regard the query as unanswerable: {recall_miss_unans}, {recall_miss_unans/len_data:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file and evaluate image predictions.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file (e.g., /map-vepfs/haozhe/xuqixin/thinking-with-documents/results/mmlongbench-doc/stage1/only_target_train_epoch1.jsonl)"
    )
    
    args = parser.parse_args()
    
    process_jsonl(args.input_file)