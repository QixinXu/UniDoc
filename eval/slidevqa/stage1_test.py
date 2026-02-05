import json
import ast
import argparse

def evaluate_predictions(file_path):
    """
    Loads a .jsonl file, evaluates model predictions against evidence pages,
    and prints evaluation metrics.
    """
    try:
        with open(file_path) as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return

    format_error = 0
    exact_match = 0
    recall_all = 0
    recall_miss = 0
    recall_miss_unans = 0

    for item in data:
        pages = item.get('evidence_pages', [])
        pred = item.get('pred', '')
        
        evidence_pages = [f'image_{str(page)}' for page in pages]
        
        # Extract content between <answer> tags
        final_answer = pred.split('<answer>')[-1].split('</answer>')[0].strip()
        
        try:
            # Safely evaluate the string to a Python list
            pred_target_images = ast.literal_eval(final_answer)
            if not isinstance(pred_target_images, list):
                raise TypeError
        except (ValueError, SyntaxError, TypeError):
            format_error += 1
            continue

        # Compare predicted pages with ground truth evidence pages
        if set(evidence_pages).issubset(set(pred_target_images)):
            if set(pred_target_images) == set(evidence_pages):
                exact_match += 1
            recall_all += 1
        else:
            recall_miss += 1
            if pred_target_images == []:
                recall_miss_unans += 1
            
    len_data = len(data)
    if len_data == 0:
        print("No data to process.")
        return

    print(f"Total samples: {len_data}")
    print(f"Format error: {format_error}, {format_error/len_data:.2%}")
    print(f"Exact match: {exact_match}, {exact_match/len_data:.2%}")
    print(f"Recall all: {recall_all}, {recall_all/len_data:.2%}")
    print(f"Recall miss: {recall_miss}, {recall_miss/len_data:.2%}")
    print(f"Recall miss due to regard the query as unanswerable: {recall_miss_unans}, {recall_miss_unans/len_data:.2%}")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate model predictions for document evidence retrieval.")
    
    # Add the file_path argument
    parser.add_argument("file_path", type=str, help="Path to the input .jsonl file.")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Run the evaluation
    evaluate_predictions(args.file_path)