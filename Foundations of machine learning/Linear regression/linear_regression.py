import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 5 * X + np.random.randn(100, 1)

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X)

# Add bias term 1 to X for dot product
X_b = np.c_[np.full((m, 1),1), X]

# Initialize parameters randomly
hypothesis = np.random.randn(2, 1)

# Gradient Descent
for iteration in range(n_iterations):
    # Use the partial derivatives of the cost function with respect to w and b respectively
    # The T in X_b.T is needed to align X_b for the dot product
    gradients = 2/m * X_b.T.dot(X_b.dot(hypothesis) - y)
    hypothesis -= learning_rate * gradients
    print(gradients)
    exit()

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(hypothesis)
# Plotting the data points and regression line
plt.scatter(X, y, color='blue', s=30)
plt.plot(X_new, y_predict, "r-")
plt.title("Linear Regression with Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.show()