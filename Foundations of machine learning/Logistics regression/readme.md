## Purpose
To predict the probability that a given instance belongs to a particular category.

### Code from scratch
~~~python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(42)
age = np.random.uniform(20, 80, 100)
exceptions = np.random.normal(0, 10, 100)
# sick = sick = (age + exceptions > 50).astype(int)
sick = (age > 50).astype(int)

# 2. Logistic Regression from scratch

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

def logistic_regression(X, y, lr=0.01, epochs=2000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for epoch in range(epochs):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)
        
        # Gradient Descent
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        weights -= lr * dw
        bias -= lr * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {compute_loss(y, y_pred)}")
    
    return weights, bias

# Adding bias term by appending a column of ones to our features
X = np.c_[np.ones(age.shape[0]), age]
y = sick

weights, bias = logistic_regression(X, y)

# 3. Visualization
plt.scatter(age, sick, c=sick, cmap='rainbow', label='Data')
plt.xlabel('Age')
plt.ylabel('Sick')

# Decision Boundary
ages_range = np.linspace(20, 80, 1000)
probabilities = sigmoid(np.dot(np.c_[np.ones(ages_range.shape[0]), ages_range], weights) + bias)
plt.plot(ages_range, probabilities, color='black', label='Decision Boundary')

plt.legend()
plt.title('Logistic Regression from Scratch')
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/95d0fe93-1745-4bac-9cb0-73d62828d691)

### Code using a library
~~~python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Plotting the data points
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')

# Plotting the logistic regression curve
X_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
probabilities = model.predict_proba(X_values)[:, 1]
plt.plot(X_values, probabilities, color='green', label='Logistic Regression Curve')

plt.axhline(0.5, color='black', linestyle='--', label='Decision Boundary')
plt.title('Logistic Regression Visualization')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/dfe420d0-ee0d-4298-b88b-19f65949a824)

