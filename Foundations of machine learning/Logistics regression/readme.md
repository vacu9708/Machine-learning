## Purpose
To predict the probability that a given instance belongs to a particular category.

### Code from scratch
~~~python
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(42)
age = np.random.uniform(20, 80, 100)
sick = (age > 50).astype(int)
# exceptions = np.random.normal(0, 10, 100)
# sick = (age + exceptions > 50).astype(int)

# 2. Logistic Regression from scratch

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

def logistic_regression(X, y, lr=0.01, epochs=50000):
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
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {compute_loss(y, y_pred)}")
    print(f"Epoch {epoch}, Loss: {compute_loss(y, y_pred)}")
    
    return weights, bias

# Adding bias term by appending a column of ones to our features
X = np.c_[np.ones(age.shape[0]), age]
y = sick

weights, bias = logistic_regression(X, y)

# 3. Visualization
ages_range = np.linspace(20, 80, 1000)
probabilities = sigmoid(np.dot(np.c_[np.ones(ages_range.shape[0]), ages_range], weights) + bias)
# Find the age where the probability crosses 0.5 (Decision Boundary)
decision_boundary_age = ages_range[np.abs(probabilities - 0.5).argmin()]

plt.scatter(age, sick, c=sick, cmap='rainbow', label='Data')
plt.xlabel('Age')
plt.ylabel('Sick')
plt.plot(ages_range, probabilities, color='black', label='Predicted Probabilities')
plt.axvline(x=decision_boundary_age, color='red', linestyle='--', label='Decision Boundary')
plt.legend()
plt.title('Logistic Regression with Decision Boundary')
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/0fa8d7ba-5098-4fce-9754-97333f398007)

### Code using a library
~~~python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, n_samples=100)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/8f99efbe-993b-4302-a323-123eb24f792b)
