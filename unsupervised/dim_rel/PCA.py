import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        Constructor of the class.

        Args:
        - n_components(int): the number of principal components to keep.

        Returns:
        - n_components(int): the number of principal components to keep.
        - components(array-like): shape (n_components, n_features), the principal components.
        - mean(array-like): shape (n_features,), the mean of each feature in the training data.
        """

        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fits the PCA model to the data.

        Args:
        - X(matrix), shape (n_samples, n_features), the training data.

        Returns:
        None
        """
            
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X):
        """
        Transforms the input data into the principal component space.

        Args:
        - X(matrix): shape (n_samples, n_features), the input data to be transformed.
aaaaa
        Returns:
        - X_transformed(matrix), shape (n_samples, n_components), the transformed data in the principal component space.
        """

        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed
    

    def fit_transform(self, X):
        """
        Fits the PCA model to the data and transforms it in one step.

        Args:
        - X(matrix): shape (n_samples, n_features), the input data.

        Returns:
        - X_transformed(matrix): shape (n_samples, n_components), the transformed data in the principal component space.
        """
            
        # Fit PCA to the data and transform it in one step
        self.fit(X)
        return self.transform(X)