import os
import math
import json
from PIL import Image
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count
import ast

# --- 图像拼接函数保持不变 ---
def concat_images(image_list, doc_id, concat_num=5, column_num=3):
    """
    拼接图像列表中的图片，并将结果保存。
    """
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()
    # 确保 image_list 不为空
    if not image_list:
        return []
        
    image_dir = os.path.dirname(image_list[0])  # Assuming all images are in the same directory
    
    for i in range(0, len(image_list), interval):
        image_id = f"{doc_id}_concat_new_{concat_num}_{i//interval}"
        image_path = os.path.join(image_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            concatenated_image_list.append(image_path)
            continue
        
        # 检查文件是否已存在，避免重复处理
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            
            # 计算总高度
            if column_num == 1:
                total_height = images_this_batch[0].height * len(images_this_batch)
            else:
                # 计算需要的行数: (元素总数 - 1) // 列数 + 1
                num_rows = (len(images_this_batch) - 1) // column_num + 1
                total_height = images_this_batch[0].height * num_rows

            # 创建新的拼接图片
            concatenated_image = Image.new('RGB', (images_this_batch[0].width * column_num, total_height), 'white')
            
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                
                # 检查是否需要换行
                if (cnt + 1) % column_num == 0:
                    y_offset += image.height
                    x_offset = 0
                    
            concatenated_image.save(image_path)
            
        concatenated_image_list.append(image_path)

    return concatenated_image_list

# --- 封装处理单个 item 的函数 ---
def process_item(item):
    """
    处理单个数据项，包括参数计算和图像拼接。
    这个函数将被多进程池调用。
    """
    try:
        images = item['images']
        doc_id = item['doc_id']

        if not images:
            return None # 如果没有图片，跳过

        if len(images) <= 20:
            # 随机确定 column_num
            random_ = random.random()
            if random_ < 0.6:
                column_num = 2
            elif random_ < 0.8:
                column_num = 3
            else:
                column_num = 4
            concat_num = math.ceil(len(images) / column_num)
        else:
            concat_num = 20
            column_num = 3

        # 调用拼接函数
        concatenated_images = concat_images(images, doc_id, concat_num=concat_num, column_num=column_num)
        
        # 将结果添加到 item 并返回
        item['concatenated_images'] = concatenated_images
        evidence_pages = item.get('evidence_pages', '[]')
        if evidence_pages != '[]':
            return None
        page_nums = ast.literal_eval(evidence_pages)
        image_per_page = math.ceil(len(images) / len(concatenated_images))
        concat_relevant_page_nums = []
        for page_num in page_nums:
            concat_index = (page_num-1) // image_per_page + 1
            concat_relevant_page_nums.append(f"image_{concat_index}")
        item['concat_relevant_page_num'] = list(set(concat_relevant_page_nums))

        return item
        
    except Exception as e:
        # 发生错误时，打印错误信息并返回 None
        print(f"Error processing docId {item.get('docId', 'Unknown')}: {e}")
        return None

# --- 主执行逻辑 ---
if __name__ == '__main__':
    data_path = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/data.json'
    output_path = '/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/unanswerable_data_concat_new.json'

    # 1. 加载数据
    with open(data_path, 'r') as f:
        # 仅加载前 4 个用于测试或调试，实际运行时可以去掉 [ : 4 ]
        data = json.load(f) 
    
    print(f"Total items to process: {len(data)}")

    # 2. 设置并启动多进程池
    # 确定要使用的进程数，通常设置为 CPU 核心数
    num_processes = cpu_count() - 1 if cpu_count() > 1 else 1 
    print(f"Using {num_processes} processes for acceleration.")

    # 使用 Pool 管理进程
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_item, data)
        
        # 收集结果并显示进度条
        new_results = []
        for result in tqdm(results_iterator, total=len(data)):
            if result is not None:
                new_results.append(result)

    print(f"Successfully processed and collected {len(new_results)} items.")
    
    # 3. 保存结果
    with open(output_path, 'w') as f:
        json.dump(new_results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_path}")