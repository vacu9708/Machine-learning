# Regression analysis
Regression analysis is a way of estimating the relationship between a dependent variable and one or more independent variables, which is used for prediction.

# Linear regression
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/856cd7af-8bc0-4a81-9328-c976e91aa969)

Linear regression is used for predicting a continuous outcome variable (also known as the dependent variable) from one or more predictor variables (also known as independent variables or features).<br>
The objective of linear regression is to find the best-fitting straight line that accurately predict the output values.<br>
A function that is set up to predict the output values is called `H`ypothesis:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/3247fc32-1810-48f7-9dc2-c00dedef1659)<br>

## Cost function and learning
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/1474a959-44e2-4d21-b9bb-66556668fd4d)<br>
The cost function means the error or discrepancy between the predicted values and the actual values in a model.<br>
The task of finding the parameters(w and b) that minimize the cost function is called **learning** or **optimization**. Optimization algorithms include the **gradient descent**.

## Cost function for linear regression
The cost function for linear regression is the Mean Squared Error (MSE), and it is given by:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/f7fe5bef-ed8b-49d4-b920-1aa1634ba459)<br>
#### Where:
- n is the number of training examples
- y is the actual output for the i-th input
- H is the predicted output for the i-th input

## Gradient descent on MSE
### 1. Numerical gradient
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/0fd173a7-1d5f-47b9-b8bd-2d27548be5fe)<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/e08e707d-66f4-45ef-a9f7-fdaf23595e31)

### 2. Derivative

## Multiple Linear Regression:
When there are more than one independent variables, the equation becomes:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/c9f8ec14-ca79-423a-82b4-3f5b86c27a09)

# Example code using numerical gradient
~~~python
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 5 * X + np.random.randn(100, 1)

# Hyperparameters
learning_rate = 0.1
n_iterations = 100
m = len(X)

# Add bias term to X
X_b = np.c_[X, np.full((m, 1),1)]

# Initialize parameters randomly
hypothesis = np.random.randn(2, 1)

# Define cost function
def cost_function(hypothesis):
    return np.sum(np.square(X_b.dot(hypothesis) - y)) / m

# Numerical gradient function
def numerical_gradient(f, x, h=1e-5):
    grads = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i][j]
            # f(x+h) calculation
            x[i][j] = tmp + h
            fxh1 = f(x)
            
            # f(x-h) calculation
            x[i][j] = tmp - h
            fxh2 = f(x)
            
            grads[i][j] = (fxh1 - fxh2) / (2*h)
            x[i][j] = tmp
    return grads

# Gradient Descent using numerical gradient
for iteration in range(n_iterations):
    gradients = numerical_gradient(cost_function, hypothesis)
    hypothesis -= learning_rate * gradients

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[X_new, np.ones((2, 1))]
y_predict = X_new_b.dot(hypothesis)

# Plotting the data points and regression line
plt.scatter(X, y, color='blue', s=30)
plt.plot(X_new, y_predict, "r-")
plt.title("Linear Regression with Numerical Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
~~~
## Result
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/5cfd9b48-70ff-4038-9917-11a9fa973d8b)

# Example code using the derivative
~~~python
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

# Add bias term to X
X_b = np.c_[np.full((m, 1),1), X]

# Initialize parameters randomly
hypothesis = np.random.randn(2, 1)

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(hypothesis) - y) # Found by taking the partial derivative of the cost function
    hypothesis -= learning_rate * gradients

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
~~~
