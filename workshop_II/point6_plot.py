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

# Set colors for each dataset
colors = ['blue', 'green', 'red', 'purple']

# Plot each dataset
for i, (title, data) in enumerate(datasets, 1):
    X, y = data
    plt.figure(i, figsize=(8, 6))  # Set figure size
    plt.scatter(X[:, 0], X[:, 1], s=30, color=colors[i-1], label=title)  # Adjust marker size and color
    plt.title(title, fontsize=16)  # Set title and fontsize
    plt.xlabel('Feature 1', fontsize=12)  # Set xlabel and fontsize
    plt.ylabel('Feature 2', fontsize=12)  # Set ylabel and fontsize
    plt.xticks(fontsize=10)  # Set xticks fontsize
    plt.yticks(fontsize=10)  # Set yticks fontsize
    plt.legend(fontsize=12)  # Add legend and set fontsize

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

print("Noisy Circles: This dataset consists of two interleaving half circles. It's not linearly separable due to noise, but it's relatively well-defined. \n" \
    "Noisy Moons: Similar to the previous dataset, but with two half moons. Again, it's not linearly separable due to noise. \n" \
    "Blobs: This dataset contains three clusters of points, each resembling a blob. It's relatively well-separated and easy to distinguish. \n" \
    "No Structure: This dataset appears to be random points scattered across the plot. It doesn't exhibit any clear structure or clusters.")

