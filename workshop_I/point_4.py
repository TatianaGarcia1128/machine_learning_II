#import modules
from unsupervised.dim_rel.SVD import SVD
from PIL import Image
import os
import numpy as np
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt

path_processed_photo ="/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/tatiana_garcia.jpg"
vector_my_face =  Image.open(os.path.join(path_processed_photo))

#Instance of SVD
param_value_singular=100
svd_instance = SVD(vector_my_face, param_value_singular)
#Use of method
imagen_recons =svd_instance.svd()

path_approximation_photo ="/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/machine_learning_II/valores_singulares_"+str(param_value_singular)+".jpg"


# Plot and save the reconstructed image
plt.imshow(imagen_recons, cmap='gray')
titulo = "valores_singulares_" + str(param_value_singular)
plt.title(titulo)
plt.savefig(f'{titulo}.jpg')
plt.show()


def different_my_photo_approximation(path_processed_photo, path_approximation_photo):
    # Obtener las imágenes originales y aproximadas (ejemplo)
    vector_my_face =  Image.open(os.path.join(path_processed_photo)).convert('L').resize((256,256))
    vector_face_approximation =  Image.open(os.path.join(path_approximation_photo)).convert('L').resize((256,256))

    imagen_original = np.array(vector_my_face).astype(np.float32)
    imagen_aproximada = np.array(vector_face_approximation).astype(np.float32)

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(imagen_original, imagen_aproximada)

    print("Error cuadrático medio (MSE):", mse)


different_my_photo_approximation(path_processed_photo, path_approximation_photo)