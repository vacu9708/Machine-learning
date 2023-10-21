# Logistic regression
Logistic Regression is used for binary classification tasks - where the goal is to predict categorical labels that are either one thing or another (e.g., spam or not spam).

### 1. Binary outcome
Logistic regression is a type of regression analysis that is well-suited for predicting the probability of an event when the response variable is categorical. In particular, it's often used for binary classification tasks, where the goal is to categorize instances into one of two classes.

## Purpose
To predict the probability that a given instance belongs to a particular category.

### Code from scratch
~~~python
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0)
age = np.random.uniform(20, 80, 100)  # Random ages between 20 and 80
sick = (age > 50).astype(int)  # Sick if age > 50

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
        gradient = np.dot(X.T, (hypothesis - y)) / m
        hypothesis_params -= learning_rate * gradient

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {compute_cost(y, hypothesis)}")
            
    print(f"Epoch {epoch}, Loss: {compute_cost(y, hypothesis)}")
    return hypothesis_params

# Prepare data
X = np.c_[np.ones(len(age)), age]  # Add a column of ones for the bias term
y = sick

# Train the logistic regression model
num_iterations = 20000
learning_rate = 0.01
hypothesis_params = logistic_regression(X, y, num_iterations, learning_rate)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(age, sick, color='blue', label='Data')
x_values = np.linspace(20, 80, 100)
x_test = np.c_[np.ones(100), x_values]  # Add a column of ones for the bias term
y_values = sigmoid(np.dot(x_test, hypothesis_params))
plt.plot(x_values, y_values, color='red', label='Probability curve')
plt.axvline(x=-hypothesis_params[0]/hypothesis_params[1], color='green', linestyle='--', label='Decision boundary')
plt.xlabel('Age')
plt.ylabel('Sick')
plt.legend()
plt.show()
~~~
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/3789a675-7ecf-49ce-b464-f7ebc9cd0fb6)

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
