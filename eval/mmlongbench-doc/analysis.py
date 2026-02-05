import jsonlines

with jsonlines.open('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/testset.jsonl') as reader:
    data = [obj for obj in reader]

target_nums = []
for item in data:
    target_nums.append(len(item['images']))

print(f"Average number of images: {sum(target_nums)/len(target_nums)}")
print(f"Max number of images: {max(target_nums)}")
print(f"Min number of images: {min(target_nums)}")

# plot the histogram
import matplotlib.pyplot as plt
plt.hist(target_nums, bins=range(1, max(target_nums) + 2), align='left', edgecolor='black')
plt.xlabel('Number of Images')
plt.ylabel('Frequency') 
plt.title('Distribution of Images in MMLongBench-Doc Test Set')
plt.xticks(range(1, max(target_nums) + 1, 10))
plt.grid(axis='y', alpha=0.75)
plt.savefig('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/image_distribution.png')
             