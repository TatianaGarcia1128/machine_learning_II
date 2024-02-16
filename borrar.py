#theta_expanded = np.expand_dims(theta, axis=1)
# np.ones((y, 1))

#X = X.data.astype('float32') / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
y = y.astype('int')

print('X', X)
print('y', y)
#y_expanded = np.expand_dims(y, axis=1)
#print('y_expanded', y_expanded.shape)

#print('tetha_expanded', theta_expanded.shape)
# Aplanar la matriz X
m = len(y)
X_flat = X.reshape(m, -1)

n_features = X_flat.shape[1]
theta = np.zeros(n_features + 1)  
print('tetha', theta.shape)
# Dividir los datos en conjuntos de entrenamiento y prueba
theta = rl_instance.train_logistic_regression(X_flat, y, theta)

# Aplicar PCA para reducir la dimensionalidad a 2 características
pca = PCA(n_components=2)

# Realizar predicciones en el conjunto de datos de entrenamiento
y_pred_train = rl_instance.predict(X_flat, theta)
y_pred_train = np.round(y_pred_train)  # Redondear las predicciones a 0 o 1

# Evaluar el rendimiento del modelo
accuracy = np.mean(y_pred_train == y) * 100
print("Precisión en el conjunto de entrenamiento:", accuracy, "%")

# Calcular las predicciones del modelo
y_true = y  # Etiquetas reales
y_pred = y_pred_train  # Predicciones del modelo

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

import seaborn as sns

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y, y_pred_train)

# Mostrar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()