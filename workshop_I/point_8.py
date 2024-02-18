import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unsupervised.dim_rel.PCA_robust import RobustPCA
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
rpca = RobustPCA(n_components=2)
rpca.fit(np.array(X_train_scaled))
X_transformed = rpca.transform(X_train_scaled)


# Train logistic regression models
log_reg_pca = LogisticRegression(max_iter=1000, C=0.1, solver='saga')
log_reg_pca.fit(X_transformed, y_train)

# Evaluate the performance of each model
y_pred_pca = log_reg_pca.predict(rpca.transform(X_test))
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print("Accuracy with PCA:", accuracy_pca)

# Plot the reduced features
plt.figure(figsize=(15, 5))

plt.subplot(1, 1, 1)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train, cmap=plt.cm.tab10)
plt.title('PCA')

plt.tight_layout()
plt.show()
