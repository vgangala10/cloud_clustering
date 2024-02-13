import json
import matplotlib.pyplot as plt
from collections import Counter

path = '/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_60/kmeans/embedding_50_classified_clustering.json'
with open(path, 'r') as file:
    cluster_classified = json.load(file)
def classified_plot(n_cluster, cluster_classified):
    labels = ['Open-cellular_MCC', 'Disorganized_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
    data = {}
    for i in range(len(labels)):
        data[labels[i]] = Counter(cluster_classified[str(n_cluster)][i])
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

    axes = axes.flatten()
    
    # Plot bar plots for each category
    for idx, (label, counter) in enumerate(data.items()):
        values = list(counter.values())
        labels = list(counter.keys())
    
        axes[idx].bar(labels, values, color='skyblue')
        axes[idx].set_title(label)
        axes[idx].set_xlabel('Values')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(range(n_cluster))

    plt.suptitle('Classification of labelled clouds for k = '+str(n_cluster), fontsize=16)

    plt.tight_layout()

    plt.show()
    
    