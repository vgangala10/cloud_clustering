import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import joblib

def plot_silhouette(X, cluster_range, n_file):
    """
    Plot silhouette plots for different numbers of clusters.

    Parameters:
    - X: Data matrix (n_samples, n_features).
    - cluster_range: Range of cluster numbers to evaluate.

    Returns:
    - None (plots the silhouette plots).
    """
    for n_clusters in cluster_range:
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax[0].set_xlim([-0.1, 1])
        ax[0].set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility
        clusterer = joblib.load(f'/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/kmeans_50_multi/{n_file}_files/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')
        cluster_labels = clusterer.predict(X)

        # Compute the silhouette scores for each sample
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax[0].fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax[0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax[0].set_title("The silhouette plot for the various clusters.")
        ax[0].set_xlabel("The silhouette coefficient values")
        ax[0].set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax[0].axvline(x=silhouette_avg, color="red", linestyle="--")

        ax[0].set_yticks([])  # Clear the yaxis labels / ticks
        ax[0].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax[1].scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        ax[1].scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax[1].scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax[1].set_title("The visualization of the clustered data.")
        ax[1].set_xlabel("Feature space for the 1st feature")
        ax[1].set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

        plt.savefig(f"/storage/climate-memmap/kmeans_50_multi/{n_file}_files/silhouette_{n_file}.png")
        plt.show()

def plot_inertia(X, cluster_range, n_file):
    """
    Plot the inertia (sum of squared distances to the nearest centroid) for different numbers of clusters.

    Parameters:
    - X: Data matrix (n_samples, n_features).
    - cluster_range: Range of cluster numbers to evaluate.

    Returns:
    - None (plots the inertia graph).
    """
    inertias = []
    for n_clusters in cluster_range:
        model = joblib.load(f'/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/kmeans_50_multi/{n_file}_files/kmeans_clustering_model_n-clusters_'+str(n_clusters)+'.joblib')
        # model.fit(X)
        inertias.append(model.inertia_)
    
    plt.plot(cluster_range, inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(cluster_range)
    plt.savefig(f"/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/kmeans_50_multi/{n_file}_files/inertia_{n_file}.png")
    plt.show()

# Example usage:
# Assuming you have your data X and a range of cluster numbers to evaluate
X = np.memmap('/storage/climate-memmap/models/ResNet34/embedding_50_transform_files_90/embeddings_50/embeddings89.memmap', dtype='float32', mode='r', shape=(10000, 50))
cluster_range = range(4, 25)  # Evaluate clusters from 4 to 25
for n_file in range(1,11):
    plot_silhouette(X, cluster_range, n_file)
    plot_inertia(X, cluster_range, n_file)
