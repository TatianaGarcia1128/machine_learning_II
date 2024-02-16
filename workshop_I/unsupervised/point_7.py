import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler


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

# Aplanar las imágenes
X_flat = X.reshape((X.shape[0], -1))

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
