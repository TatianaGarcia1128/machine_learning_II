import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from clustering.logistic_regression import LOGISTIC_REGRESSION
import numpy as np

#Instanciar clase de regresión logística
rl_instance = LOGISTIC_REGRESSION()

#Cargar los datos MNIST
mnist = fetch_openml('mnist_784', parser='auto')

#Preprocesamiento de datos
X = mnist.data.astype('float32') / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
y = mnist.target.astype('int')


X_08 = X[(y == 0) | (y == 8)]
y_08 = y[(y == 0) | (y == 8)]
y_08 = np.where(y_08 == 0, 0, 1)  # Convertir etiquetas 0 a 0 y etiquetas 8 a 1

#Inicialización de parámetros
n_features = X_08.shape[1]
theta = np.zeros(n_features + 1)  # Inicializar los parámetros, incluyendo el término de sesgo (bias)

theta = rl_instance.train_logistic_regression(X_08, y_08, theta)

# Realizar predicciones en el conjunto de datos de entrenamiento
y_pred_train = rl_instance.predict(X_08, theta)
y_pred_train = np.round(y_pred_train)  # Redondear las predicciones a 0 o 1

# Evaluar el rendimiento del modelo
accuracy = np.mean(y_pred_train == y_08) * 100
print("Precisión en el conjunto de entrenamiento:", accuracy, "%")

# Calcular las predicciones del modelo
y_true = y_08  # Etiquetas reales
y_pred = y_pred_train  # Predicciones del modelo

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Mostrar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(set(y_true)))
plt.xticks(tick_marks, ['0', '1'], rotation=45)
plt.yticks(tick_marks, ['0', '1'])
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')



# Anotar los valores de la matriz de confusión
thresh = conf_matrix.max() / 2.
for i, j in [(i, j) for i in range(conf_matrix.shape[0]) for j in range(conf_matrix.shape[1])]:
    plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()