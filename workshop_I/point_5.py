#import modules
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
digits = load_digits()

# Filter the dataset to only include digits 0 and 8
filters = (digits.target == 0) | (digits.target == 8)
X = digits.images[filters]
y = digits.target[filters]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Flatten the data into a two-dimensional array
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Baseline logistic regression accuracy:", accuracy)
print("The baseline performance of this naive logistic regression model can vary depending on factors like the quality of features (raw pixel values in this case), the chosen model, " \
    "and hyperparameters. However, we can make some general observations: \n" \
    "Simplicity of the model: Logistic regression is a simple linear classifier. It may struggle to capture complex patterns present in raw images. \n"
    "Baseline performance: Given that logistic regression is a simple model and we are using raw pixel values as features without any preprocessing, we can expect a moderate baseline " \
    "performance. The accuracy typically ranges from around 85% to 95% on this binary classification task (0s vs. 8s), but this can vary. \n" \
    "Improvement potential: The baseline performance provides a starting point for further experimentation. Techniques such as feature scaling, dimensionality reduction, or more complex " \
    "odels (e.g., deep neural networks) can potentially improve performance beyond this baseline.")
