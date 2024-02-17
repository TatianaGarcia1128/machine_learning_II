import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist['data'], mnist['target'].astype(np.uint8)

# Filter dataset to include only 0s and 8s
X = X[(y == 0) | (y == 8)]
y = y[(y == 0) | (y == 8)]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Dimensionality reduction using SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_train)

# Dimensionality reduction using t-SNE
perplexity_value = min(30, X_train.shape[0] - 1)
tsne = TSNE(n_components=2, perplexity=perplexity_value)
X_tsne = tsne.fit_transform(X_train)

# Train logistic regression models
log_reg_pca = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_pca.fit(X_pca, y_train)

log_reg_svd = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_svd.fit(X_svd, y_train)

log_reg_tsne = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_tsne.fit(X_tsne, y_train)

# Evaluate the performance of each model
y_pred_pca = log_reg_pca.predict(pca.transform(X_test))
accuracy_pca = accuracy_score(y_test, y_pred_pca)

y_pred_svd = log_reg_svd.predict(svd.transform(X_test))
accuracy_svd = accuracy_score(y_test, y_pred_svd)

perplexity_value = min(30, X_test.shape[0] - 1)
tsne = TSNE(n_components=2, perplexity=perplexity_value)
y_pred_tsne = log_reg_tsne.predict(tsne.fit_transform(X_test))
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)

print("Accuracy with PCA:", accuracy_pca)
print("Accuracy with SVD:", accuracy_svd)
print("Accuracy with t-SNE:", accuracy_tsne)

# Plot the reduced features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.tab10)
plt.title('PCA')

plt.subplot(1, 3, 2)
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y_train, cmap=plt.cm.tab10)
plt.title('SVD')

plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap=plt.cm.tab10)
plt.title('t-SNE')

plt.tight_layout()
plt.show()
