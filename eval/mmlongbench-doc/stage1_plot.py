import matplotlib.pyplot as plt

def plot_training_metrics():

    epochs = [0, 1, 2, 3, 4, 5]
    exact_match = [37.78, 48.15, 50.37, 45.93, 46.67, 46.67]
    recall_all = [49.63, 62.22, 60.00, 57.04, 56.30, 56.30]
    recall_miss = [49.63, 37.78, 40.00, 42.96, 43.70, 43.70]

    exact_match1 = [37.78, 51.85, 48.15, 44.44, 45.93, 45.93]
    recall_all1 = [49.63, 65.19, 62.96, 58.52, 57.78, 57.04]
    recall_miss1 = [49.63, 34.81, 37.04, 41.48, 40.74, 42.96]

    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, exact_match, marker='o', linestyle='-', color='b', label='Exact Match target only')
    plt.plot(epochs, recall_all, marker='s', linestyle='-', color='g', label='Recall All target only')
    plt.plot(epochs, recall_miss, marker='^', linestyle='-', color='r', label='Recall Miss target only')

    plt.plot(epochs, exact_match1, marker='o', linestyle='--', color='b', label='Exact Match every image')
    plt.plot(epochs, recall_all1, marker='s', linestyle='--', color='g', label='Recall All every image')
    plt.plot(epochs, recall_miss1, marker='^', linestyle='--', color='r', label='Recall Miss every image')
    
    plt.axhline(y=3.70, color='b', linestyle='-.', label='FRAG Exact Match')
    plt.axhline(y=54.81, color='g', linestyle='-.', label='FRAG Recall ALL')
    plt.axhline(y=45.19, color='r', linestyle='-.', label='FRAG Recall Miss')  

    plt.axhline(y=29.63, color='b', linestyle='--', label='Chunk 5 prompt Exact Match')
    plt.axhline(y=60.00, color='g', linestyle='--', label='Chunk 5 prompt Recall All')
    plt.axhline(y=40.00, color='r', linestyle='--', label='Chunk 5 prompt Recall Miss')

    plt.title('Performance Metrics Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    
    plt.grid(True)
    plt.legend()
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig('/map-vepfs/haozhe/xuqixin/thinking-with-documents/data/mmlongbench-doc/only_target_train_performance_metrics.png')


if __name__ == "__main__":
    plot_training_metrics()