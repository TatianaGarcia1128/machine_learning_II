import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
from dim_rel.PCA import PCA
from dim_rel.SVD import SVD
from dim_rel.t_SNE import t_SNE
from clustering.logistic_regression import LOGISTIC_REGRESSION

#Cargar y Preprocesar las Imágenes
def load_images(path_cohort):
    images = []
    etiquetas = []
    #Get the list of image file names in the directory
    list_images = [file for file in os.listdir(path_cohort) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    cont=0
    for file in list_images:
          # Obtener el nombre del archivo y su extensión
        file_name, file_extension = os.path.splitext(file)
        
        # Si la extensión es .png, cambiarla a .jpg
        if file_extension.lower() == '.png':
            new_file_name = file_name + '.jpg'
            os.rename(os.path.join(path_cohort, file), os.path.join(path_cohort, new_file_name))
        else:
            new_file_name = file
            
        #Image reading
        image_cohort = Image.open(os.path.join(path_cohort, new_file_name)).convert('L').resize((256, 256))

        etiquetas.append(cont)
        cont+=1
        images.append(np.array(image_cohort, dtype=np.float32) / 255.0)   # Aplanar la imagen y agregarla a la lista
    
    return np.array(images), np.array(etiquetas)


folder_path = "/Users/tatianagarcia/Library/Mobile Documents/com~apple~CloudDocs/Especialización ciencia de datos y analítica/Machine Learning 2/photos_group"
X, y = load_images(folder_path)
#X = X.data.astype('float32') / 255.0 
# Aplanar las imágenes
X = X.reshape((X.shape[0], -1))

rl_instance = LOGISTIC_REGRESSION()
svd = SVD(X, 2)
tsne = t_SNE(X)


#Inicialización de parámetros
n_features = X.shape[1]
theta = np.zeros(n_features + 1)  # Inicializar los parámetros, incluyendo el término de sesgo (bias)

theta = rl_instance.train_logistic_regression(X, y, theta)

# Split the dataset into train and test sets
# Realizar predicciones en el conjunto de datos de entrenamiento
y_pred_train = rl_instance.predict(X, theta)
y_pred_train = np.round(y_pred_train)  # Redondear las predicciones a 0 o 1

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(y_pred_train)

# Dimensionality reduction using SVD
svd = svd(n_components=2)
X_svd = svd.fit_transform(y_pred_train)

# Dimensionality reduction using t-SNE
perplexity_value = min(30, y_pred_train.shape[0] - 1)
tsne = tsne(n_components=2, perplexity=perplexity_value)
X_tsne = tsne.fit_transform(y_pred_train)

# Train logistic regression models
log_reg_pca = rl_instance(max_iter=1000, C=0.1, solver='saga')
log_reg_pca.fit(X_pca, y_pred_train)

log_reg_svd = rl_instance(max_iter=1000, C=0.1, solver='saga')
log_reg_svd.fit(X_svd, y_pred_train)

log_reg_tsne = rl_instance(max_iter=1000, C=0.1, solver='saga')
log_reg_tsne.fit(X_tsne, y_pred_train)

# Plot the reduced features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_train, cmap=plt.cm.tab10)
plt.title('PCA')

plt.subplot(1, 3, 2)
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y_pred_train, cmap=plt.cm.tab10)
plt.title('SVD')

plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_train, cmap=plt.cm.tab10)
plt.title('t-SNE')

plt.tight_layout()
plt.show()
