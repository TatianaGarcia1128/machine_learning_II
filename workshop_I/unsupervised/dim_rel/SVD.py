import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class SVD:
    def __init__(self, A, s):
        """
        Constructor of the SVD class.

        Parameters:
        - A: array-like, input matrix.
        - s: int, number of singular values to consider.

        Attributes:
        - A(matrix): input matrix.
        - s(int): number of singular values to consider.
        """

        self.A = A
        self.s = s

    def svd(self):
        """
        Compute the Singular Value Decomposition of matrix A.

        :return: U, D, V matrices such that A = U * D * V^T
        """

        # Compute the Singular Value Decomposition
        U, D, V = np.linalg.svd(self.A)

        # Reconstruct the image using the first s singular values
        imagen_recons = np.matrix(U[:, :self.s]) * np.diag(D[:self.s]) * np.matrix(V[:self.s,:])

        # Plot and save the reconstructed image
        plt.imshow(imagen_recons, cmap='gray')
        titulo = "valores_singulares_" + str(self.s)
        plt.title(titulo)
        plt.savefig(f'{titulo}.jpg')
        plt.show()
