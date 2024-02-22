import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn import datasets
import matplotlib.pyplot as plt

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05) 
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

datasets = [
    ("Noisy Circles", noisy_circles),
    ("Noisy Moons", noisy_moons),
    ("Blobs", blobs),
    ("No Structure", no_structure)
]

# Define clustering algorithms
clustering_algorithms = [
    ('KMeans', KMeans(n_clusters=2, max_iter=300)),
    ('KMedoids', KMedoids(n_clusters=2, max_iter=300)),
    ('DBSCAN', DBSCAN(eps=0.3, min_samples=10)),
    ('SpectralClustering', SpectralClustering(n_clusters=2, affinity='nearest_neighbors'))
]

# Perform clustering and plot results
for i, (title, data) in enumerate(datasets, 1):
    X, y = data
    plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plt.suptitle(title, fontsize=16)
    plot_num = 1
    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(1, len(clustering_algorithms), plot_num)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
        plt.title(name, size=12)
        plot_num += 1

plt.show()