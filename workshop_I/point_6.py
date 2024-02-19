#Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from unsupervised.dim_rel.PCA import PCA
from unsupervised.dim_rel.SVD import SVD
from sklearn.linear_model import LogisticRegression
from unsupervised.dim_rel.TSNE import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
digits = load_digits()

# Filter the dataset to only include digits 0 and 8
filters = (digits.target == 0) | (digits.target == 8)
X = digits.images[filters]
y = digits.target[filters]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Flatten the data into a two-dimensional array
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.fit_transform(X_test_flat)

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Dimensionality reduction using SVD
svd = SVD(n_components=2)
X_svd = svd.fit_transform(X_train_scaled)

# Dimensionality reduction using t-SNE
tsne = TSNE(n_dimensions=2)
tsne.fit(X_train_scaled)
# transform the data using the T-SNE object
X_transformed_tsne = tsne.transform(X_train_scaled,1000)

# Train logistic regression models
log_reg_pca = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_pca.fit(X_pca, y_train)

log_reg_svd = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_svd.fit(X_svd, y_train)

log_reg_tsne = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_tsne.fit(X_transformed_tsne, y_train)

# Evaluate the performance of each model
y_pred_pca = log_reg_pca.predict(pca.transform(X_test_scaled))
accuracy_pca = accuracy_score(y_test, y_pred_pca)

y_pred_svd = log_reg_svd.predict(svd.fit_transform(X_test_scaled))
accuracy_svd = accuracy_score(y_test, y_pred_svd)

tsne.fit(X_test_scaled)
y_pred_tsne = log_reg_tsne.predict(tsne.transform(X_test_scaled, iterations=1000))
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
plt.scatter(X_transformed_tsne[:, 0], X_transformed_tsne[:, 1], c=y_train, cmap=plt.cm.tab10)
plt.title('t-SNE')

plt.tight_layout()
plt.show()

print('The best performance using my libraries corresponds to PCA, the other models have very poor accuracy but in the graphs you can see a good separation of the data.')