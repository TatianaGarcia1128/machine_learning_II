#import modules
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np
from unsupervised.clustering.kmedoids import KMEDOIDS
#
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

#create scattered data X
X, y = make_blobs(
n_samples=500, n_features=2,
centers=4,
cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1,)

# Run the kmeans algorithm
k = 5
km = KMEDOIDS(k)
km.fit(X, k, p=2)
y_predict = km.predict(X)


score = silhouette_score(X, y_predict, metric='euclidean')

# Print the score
print('Silhouetter Score: %.3f' % score)
print('Executing the above code predicts the Silhouette score of 0.571. This indicates that the clustering is reasonably good, ' \
      'since the Silhouette Score is closer to 1 than to -1 or 0. A Silhouette Score of 0.553 suggests that the clusters are well ' \
        'separated and that the objects within each cluster are relatively homogeneous.')


range_n_clusters = [2, 3, 4, 5, 6]
silhouette_scores = []  #List to store Silhouette coefficients
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMEDOIDS(k=n_clusters)
    clusterer.fit(X, k, p=2)
    cluster_labels = clusterer.predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    #Add the coefficient to the list
    silhouette_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.medoids
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMedoids clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

#Finding the value of k with the highest Silhouette coefficient
best_k = 2 + silhouette_scores.index(max(silhouette_scores))
print('Value of k with the highest Silhouette coefficient', best_k)
print('In general, for Kmeans the clusters for k=6 visually do not seem the best; since there is a cluster below the average, data apparently misclassified ' \
      'since the silhouette coefficient is negative.')
