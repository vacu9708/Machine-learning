# Regression analysis
Regression analysis is a way of estimating the relationship between a dependent variable and one or more independent variables, which is used for prediction.

# Linear regression
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/856cd7af-8bc0-4a81-9328-c976e91aa969)

Linear regression is used for predicting a continuous outcome variable (also known as the dependent variable) from one or more predictor variables (also known as independent variables or features).<br>
The objective of linear regression is to find the best-fitting straight line that accurately predict the output values.<br>
A function that is set up to predict the output values is called `H`ypothesis:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/3247fc32-1810-48f7-9dc2-c00dedef1659)<br>

## Cost function and learning
The cost function means the error or discrepancy between the predicted values and the actual values in a model.<br>
The task of finding the parameters(w and b) that minimize the cost function is called learning or optimization algorithm. Optimization algorithms include the **gradient descent**.

## Cost function for linear regression
The cost function for linear regression is the Mean Squared Error (MSE), and it is given by:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/f7fe5bef-ed8b-49d4-b920-1aa1634ba459)<br>
#### Where:
- n is the number of training examples
- y is the actual output for the i-th input
- H is the predicted output for the i-th input

## Multiple Linear Regression:
When there are more than one independent variables, the equation becomes:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/c9f8ec14-ca79-423a-82b4-3f5b86c27a09)

# Example code using gradient descent
~~~python
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X)

# Add bias term to X
X_b = np.c_[np.ones((m, 1)), X]

# Initialize parameters (beta) randomly
beta = np.random.randn(2, 1)

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(beta) - y)
    beta -= learning_rate * gradients

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(beta)

# Plotting the data points and regression line
plt.scatter(X, y, color='blue', s=30)
plt.plot(X_new, y_predict, "r-")
plt.title("Linear Regression with Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
~~~

# Using normal equation
~~~python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term (x0 = 1) for every sample to handle the intercept
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 for all samples

# Calculate the parameters using the normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Predictions using the computed parameters
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add x0 = 1 for all samples
y_predict = X_new_b.dot(theta_best)

# Plot the results
plt.scatter(X, y, color='blue', label="True values")
plt.plot(X_new, y_predict, 'r-', label="Predicted line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Simple Linear Regression from Scratch")
plt.show()

print("y-intercept (beta_0):", theta_best[0][0])
print("Slope (beta_1):", theta_best[1][0])
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/5cfd9b48-70ff-4038-9917-11a9fa973d8b)
