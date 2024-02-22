#import modules
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
from unsupervised.clustering.kmeans import KMEANS
from unsupervised.clustering.kmedoids import KMEDOIDS
from sklearn.metrics import silhouette_score, silhouette_samples

#create scattered data X
X, y = make_blobs(
n_samples=500, n_features=2,
centers=4,
cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1,)

# # Plot the dataset
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
# plt.title('Generated Dataset')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.colorbar()
# plt.grid(True)
# plt.show()

print('There are a total of 4 clusters; only one of them is visually very separated from the other 3.')


# Calcular las distancias entre pares de centros de clusters
centros = []
for i in range(4):
    centros.append(X[y == i].mean(axis=0))

distancias_pares = np.zeros((4, 4))
for i in range(4):
    for j in range(i+1, 4):
        dist = euclidean(centros[i], centros[j])
        distancias_pares[i, j] = dist
        distancias_pares[j, i] = dist

print("Distancias entre pares de centros de clusters:")
print(distancias_pares)


# Run the kmeans algorithm
n_cluster = 4
for n_cluster in range(2, 6):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    # K-means
    kmeans = KMEANS(n_cluster, max_iter=1000)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X)
    silhouette_avg_kmeans = silhouette_score(X, kmeans_labels)
    sample_silhouette_values_kmeans = silhouette_samples(X, kmeans_labels)

    # k-medoids
    kmedoids = KMEDOIDS(n_cluster)
    kmedoids.fit(X)
    kmedoids_labels = kmedoids.predict(X)
    silhouette_avg_kmedoids = silhouette_score(X, kmedoids_labels)
    sample_silhouette_values_kmedoids = silhouette_samples(X, kmedoids_labels)


    # Plotting Silhouette Plots
    y_lower = 10
    for i in range(n_cluster):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values_kmeans = sample_silhouette_values_kmeans[kmeans_labels == i]
        ith_cluster_silhouette_values_kmedoids = sample_silhouette_values_kmedoids[kmedoids_labels == i]
        
        ith_cluster_silhouette_values_kmeans.sort()
        ith_cluster_silhouette_values_kmedoids.sort()

        size_cluster_i_kmeans = ith_cluster_silhouette_values_kmeans.shape[0]
        size_cluster_i_kmedoids = ith_cluster_silhouette_values_kmedoids.shape[0]

        y_upper_kmeans = y_lower + size_cluster_i_kmeans
        y_upper_kmedoids = y_lower + size_cluster_i_kmedoids

        color = plt.cm.nipy_spectral(float(i) / n_cluster)
        
        # Plotting k-means silhouette plot
        ax1.fill_betweenx(np.arange(y_lower, y_upper_kmeans), 0, ith_cluster_silhouette_values_kmeans, facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i_kmeans, str(i))
        y_lower = y_upper_kmeans + 10
        
        # Plotting k-medoids silhouette plot
        ax2.fill_betweenx(np.arange(y_lower, y_upper_kmedoids), 0, ith_cluster_silhouette_values_kmedoids, facecolor=color, edgecolor=color, alpha=0.7)
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i_kmedoids, str(i))
        y_lower = y_upper_kmedoids + 10

        ax1.set_title("Silhouette plot for KMeans clustering with {} clusters".format(n_cluster))
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg_kmeans, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ax2.set_title("Silhouette plot for KMedoids clustering with {} clusters".format(n_cluster))
        ax2.set_xlabel("Silhouette coefficient values")
        ax2.set_ylabel("Cluster label")
        ax2.axvline(x=silhouette_avg_kmedoids, color="red", linestyle="--")
        ax2.set_yticks([])
        ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans and KMedoids clustering on sample data "
                    "with n_clusters = %d" % n_cluster),
                    fontsize=14, fontweight='bold')

        plt.show()