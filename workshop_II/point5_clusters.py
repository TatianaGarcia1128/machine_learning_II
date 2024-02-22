#import modules
from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import numpy as np

#create scattered data X
X, y = make_blobs(
n_samples=500, n_features=2,
centers=4,
cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1,)

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title('Generated Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.grid(True)
plt.show()

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

