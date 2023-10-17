# Regression analysis
Regression analysis is a way of estimating the relationship between a dependent variable and one or more independent variables, which is used for prediction.

# Linear regression
Linear regression is used for predicting a continuous outcome variable (also known as the dependent variable) from one or more predictor variables (also known as independent variables or features).<br>
The objective of linear regression is to find the best-fitting straight line that accurately predict the output values.<br>
The function that predicts the output values is called `H`ypothesis:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/3247fc32-1810-48f7-9dc2-c00dedef1659)<br>

## Cost Function
The cost function means the error or discrepancy between the predicted values and the actual values in a model.<br>
The prediction is done by minimizing the cost function.<br>
The cost function for linear regression is the Mean Squared Error (MSE), and it is given by:<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/f7fe5bef-ed8b-49d4-b920-1aa1634ba459)<br>
#### Where:
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/a65880ae-6d5c-4e5d-9955-5c907a7b3f6d)

## Learning:
The learning process is done by adjusting the weights (coefficients) using optimization techniques like gradient descent to minimize the cost function.

## Multiple Linear Regression:
When there are more than one independent variables, the equation becomes:
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/c9f8ec14-ca79-423a-82b4-3f5b86c27a09)

# Example code using a machine learning library
~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(1)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train[0],X_test[0],y_train[0],y_test[0])
# Create a linear regression model
lin_reg = LinearRegression()

# Fit the model to the training data
lin_reg.fit(X_train, y_train)

# Print the coefficients
print("y-intercept (beta_0):", lin_reg.intercept_[0])
print("Slope (beta_1):", lin_reg.coef_[0][0])

# Predict on test data
y_pred = lin_reg.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Plot the results
plt.scatter(X_test, y_test, color='blue', label="True values")
plt.plot(X_test, y_pred, 'r.', label="Predicted values")
# plot X and y with a small dot
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Simple Linear Regression")
plt.figure(2)
plt.scatter(X, y)
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/66b3b99a-88a5-4413-98ec-2edfece4fbcb)<br>
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/e25360d2-cc6f-4941-8dd8-da0c360585eb)

# From scratch
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
