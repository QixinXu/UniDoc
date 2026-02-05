import json
import os

def split_json_file(input_file, output_dir, chunk_size=100):
    """
    Splits a large JSON file (list of dictionaries) into multiple smaller JSON files.

    Args:
        input_file (str): The path to the original JSON file.
        output_dir (str): The directory where the new JSON files will be saved.
        chunk_size (int): The number of data entries per output file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data from the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate the total number of chunks
    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    print(f"Total entries: {len(data)}")
    print(f"Splitting into {total_chunks} files of size {chunk_size}...")

    # Iterate through the data and create a new file for each chunk
    for i in range(total_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = data[start_index:end_index]

        output_filename = f"part_{i + 1}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=4)
        
        print(f"Saved {output_path}")


input_json_path = "/map-vepfs/haozhe/xuqixin/LLaMA-Factory/data/training/stage1_retrival/dude_answerable_scaled/retrieve_data_dude_every_page_scaled.json"
output_directory = "/map-vepfs/haozhe/xuqixin/LLaMA-Factory/data/training/stage1_retrival/dude_answerable_scaled/retrieve_data_dude_every_page_scaled"
chunk_size = 2000

# 调用函数
split_json_file(input_json_path, output_directory, chunk_size)