# Logistic regression
Logistic Regression is used for binary classification tasks - where the goal is to predict categorical labels that are either one thing or another (e.g., spam or not spam).

### Binary Classification:
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/dfa4a74a-15df-441c-9952-a49fbbc5e6e1)<br>
Logistic Regression estimates the probability that a given input point belongs to a certain class. The probability estimation is done through the logistic function, which is an S-shaped curve (Sigmoid function).<br>
Logistic regression can be extended to multiclass classification (called Multinomial Logistic Regression) by using techniques like one-vs-rest (OvR).

### Parameter Estimation
Logistic regression performs learning by minimizing the cost function just like linear regression.<br>
The predicted function is a linear function in linear regression while it is a sigmoid function in logistic regression.
The parameters of the logistic regression model are estimated using a method called Maximum Likelihood Estimation (MLE). MLE aims to find the parameters that maximize the likelihood of observing the given data.

### Decision Boundary:
Logistic regression creates a linear decision boundary in the feature space, separating the two classes. The decision boundary is where the estimated probability is 0.5. Observations on one side of the boundary are classified into one category, and observations on the other side are classified into the other category.

## Purpose
To predict the probability that a given instance belongs to a particular category.

### Code from scratch
~~~python
import numpy as np
import matplotlib.pyplot as plt

# Logistic regression from scratch
def compute_cost(y, hypothesis):
    return -np.mean(y * np.log(hypothesis) + (1-y) * np.log(1-hypothesis))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(X, y, epoch, learning_rate):
    m, n = X.shape
    hypothesis_params = np.zeros(n)
    for epoch in range(epoch):
        linear_hypothesis = np.dot(X, hypothesis_params)
        hypothesis = sigmoid(linear_hypothesis)
        gradients = np.dot(X.T, (hypothesis - y)) / m # (A hypothesis is put in the cost function, but the actual differentiation is skipped)
        hypothesis_params -= learning_rate * gradients

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {compute_cost(y, hypothesis)}")
            
    print(f"Epoch {epoch}, Loss: {compute_cost(y, hypothesis)}")
    return hypothesis_params

def predict(X, hypothesis_params):
    linear_hypothesis = np.dot(X, hypothesis_params)
    hypothesis = sigmoid(linear_hypothesis)
    predictions = (hypothesis > 0.5).astype(int)  # Threshold of 0.5 to classify as 0 or 1
    return predictions

# Generate training data
np.random.seed(0)
ages = np.random.uniform(20, 80, 100)  # Random ages between 20 and 80
sick = (ages > 50).astype(int)  # Sick if age > 50
X = np.c_[np.ones(len(ages)), ages]  # Add a column of ones for the bias term
y = sick

# Train the logistic regression model
num_iterations = 20000
learning_rate = 0.01
hypothesis_params = logistic_regression(X, y, num_iterations, learning_rate)

# Example predictions
test_ages = np.array([25, 55, 75])
test_X = np.c_[np.ones(len(test_ages)), test_ages]  # Add a column of ones for the bias term
predictions = predict(test_X, hypothesis_params)
for age, prediction in zip(test_ages, predictions):
    print(f"Age: {age}, Predicted Sick: {prediction}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(ages, sick, color='blue', label='Training data')
x_values = np.linspace(20, 80, 100)
X_values = np.c_[np.ones(100), x_values]  # Add a column of ones for the bias term
y_values = sigmoid(np.dot(X_values, hypothesis_params))
plt.plot(x_values, y_values, color='red', label='Probability curve')
plt.axvline(x=-hypothesis_params[0]/hypothesis_params[1], color='green', linestyle='--', label='Decision boundary')
plt.xlabel('Age')
plt.ylabel('Sick')
plt.legend()
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/fcacd957-95ea-4cad-8f91-f55d062ef0d0)<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/3deafb5a-db94-4b5b-98ce-3607107248e1)

### Code using a library
~~~python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features for visualization
y = (iris.target != 0) * 1  # convert the target to binary

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Define the sigmoid function
def sigmoid(x1, x2, theta_0, theta_1, theta_2):
    z = theta_0 + theta_1 * x1 + theta_2 * x2
    return expit(z)

# Get the coefficients from logistic regression model
theta_0 = log_reg.intercept_[0]
theta_1, theta_2 = log_reg.coef_.T

# Create a mesh grid for plotting
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get the sigmoid output for every point in the mesh grid
Z = sigmoid(xx.ravel(), yy.ravel(), theta_0, theta_1, theta_2)
Z = Z.reshape(xx.shape)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sigmoid surface
ax.plot_surface(xx, yy, Z, alpha=0.8, cmap='viridis')

# Plot the training data
ax.scatter(X[y==0][:, 0], X[y==0][:, 1], np.zeros((X[y==0].shape[0])), color='red', label='Class 0')
ax.scatter(X[y==1][:, 0], X[y==1][:, 1], np.ones((X[y==1].shape[0])), color='blue', label='Class 1')

# Decision boundary is where sigmoid = 0.5, i.e., z = 0
ax.contour(xx, yy, Z, levels=[0.5], colors='black')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Sigmoid Output')
ax.legend()
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/bbfb5182-2220-4ffa-ba9b-33151aa17f53)
