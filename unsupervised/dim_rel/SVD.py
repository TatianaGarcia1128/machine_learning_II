import numpy as np

class SVD:
    def __init__(self, n_components=None):
        """
        Constructor of the SVD class.

        Parameters:
        - A: array-like, input matrix.
        - s: int, number of singular values to consider.

        Attributes:
        - A(matrix): input matrix.
        - s(int): number of singular values to consider.
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None


    def fit(self, X):
        # Calcula la SVD
        self.U, self.S, self.Vt = np.linalg.svd(X, full_matrices=False)
        
        # Reduce a los primeros n_componentes si se especifica
        if self.n_components is not None:
            self.U = self.U[:, :self.n_components]
            self.S = np.diag(self.S[:self.n_components])
            self.Vt = self.Vt[:self.n_components, :]
    
    def fit_transform(self, X):
        self.fit(X)
        # Transforma los datos a los componentes principales
        return np.dot(self.U, self.S)

    def svd(self, A):
        """
        Compute the Singular Value Decomposition of matrix A.

        :return: U, D, V matrices such that A = U * D * V^T
        """

        # Compute the Singular Value Decomposition
        U, D, V = np.linalg.svd(A)

        # Reconstruct the image using the first s singular values
        imagen_recons = np.matrix(U[:, :self.n_components]) * np.diag(D[:self.n_components]) * np.matrix(V[:self.n_components,:])

        return imagen_recons