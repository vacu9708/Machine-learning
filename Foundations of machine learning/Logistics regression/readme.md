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

