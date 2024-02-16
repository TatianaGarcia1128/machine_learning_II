import numpy as np


class LOGISTIC_REGRESSION:
    def __init__(self):
        """
        Constructor of the Logistic Regression class.

        Initializes the z attribute to None.
        """
        self.z = None

    def sigmoid(self,z):
        """
        Compute the sigmoid function.

        Args:
        - z(array-like): input to the sigmoid function.

        Returns:
        - sigmoid_output(array-like): output of the sigmoid function.
        """

        self.z = z
        return 1 / (1 + np.exp(-self.z))

    def train_logistic_regression(self, X, y, theta, learning_rate=0.01, num_epochs=100):
        """
        Train the logistic regression model using gradient descent.

        Args:
        - X: array-like, shape (m, n), input features.
        - y: array-like, shape (m,), target labels.
        - theta: array-like, shape (n + 1,), initial parameters.
        - learning_rate: float, learning rate for gradient descent (default is 0.01).
        - num_epochs: int, number of epochs for training (default is 100).

        Returns:
        - theta: array-like, shape (n + 1,), trained parameters.
        """
                
        m = len(y)  # Number of training examples
        X_with_bias = np.hstack((np.ones((m, 1)), X))  # Add a column of ones for the bias term
        for epoch in range(num_epochs):  # Iterate over the specified number of epochs
            z = np.dot(X_with_bias, theta)  # Compute the linear combination of features and parameters
            h = self.sigmoid(z)  # Compute the sigmoid of z
            gradient = np.dot(X_with_bias.T, (h - y)) / m  # Compute the gradient of the cost function
            theta -= learning_rate * gradient  # Update parameters using gradient descent
        return theta.squeeze()  # Return the trained parameters

    def predict(self, X, theta):
        """
        Make predictions using the logistic regression model.

        Parameters:
        - X: array-like, shape (m, n), input features.
        - theta: array-like, shape (n + 1,), parameters of the model.

        Returns:
        - predictions: array-like, shape (m,), predicted labels.
        """
        X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))  # Agregar una columna de unos para el término de sesgo
        z = np.dot(X_with_bias, theta)
        return self.sigmoid(z)



