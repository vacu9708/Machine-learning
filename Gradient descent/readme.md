# Gradient Descent
Gradient Descent is a first-order iterative optimization algorithm used to minimize a function. It's widely used in machine learning and deep learning for training models.

## Basic Idea
Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves towards a set of parameter values that minimize the function. This iterative minimization is achieved by taking steps in the negative direction of the function gradient.

## Algorithm
1. `Initialization:` Start with an initial point (or a guess).
2. `Compute Gradient:` Calculate the gradient of the function at the current point.
3. `Update:` Move in the opposite direction of the gradient by a step size (often called the learning rate, α). The formula for the update is:

![image](https://github.com/vacu9708/Machine-learning/assets/67142421/6cc2f1fc-a533-4d4e-88ea-fe0baecbf874)

## Types of Gradient Descent
- `Batch Gradient Descent:` Uses the entire training set to compute the gradient at each step. It can be computationally expensive for large datasets.
- `Stochastic Gradient Descent (SGD):` Uses only one training example at each step. This introduces a lot of variance and can lead to erratic convergence, but often converges much faster.
- `Mini-batch Gradient Descent:` A compromise between batch and stochastic versions. It uses a subset of the training data at each step.

## Key Hyperparameter: Learning Rate
The learning rate, α, is a crucial hyperparameter. If it's too high, the algorithm might overshoot the minimum and diverge. If it's too low, the algorithm will converge slowly. In practice, it might be necessary to try multiple values to find the best one.

## Challenges
- Choosing a proper learning rate.
- The possibility of getting stuck in local minima (especially in high-dimensional spaces).
- Sensitive to feature scaling. It's often recommended to scale input features.

## Python Implementation
```python
def gradient_descent(gradient_func, initial_theta, learning_rate, num_iterations):
    theta = initial_theta
    for _ in range(num_iterations):
        gradient = gradient_func(theta)
        theta -= learning_rate * gradient
    return theta


## Code
~~~python
def function(x):
    return x**2

def get_gradient(x):
    return 2*x

def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    history = []  # to store the path of x during descent

    for i in range(num_iterations):
        gradient = get_gradient(x)
        x = x - learning_rate * gradient
        history.append(x)

    return x, history

import matplotlib.pyplot as plt

initial_x = 10  # start from x=10
learning_rate = 0.1
num_iterations = 50

final_x, history = gradient_descent(initial_x, learning_rate, num_iterations)

# Plotting
x = [i/10 for i in range(-100, 101)]
y = [function(i) for i in x]

plt.plot(x, y, '-r', label='y=x^2')
plt.scatter(history, [function(i) for i in history], c='blue', marker='o')

plt.title('Gradient Descent on y=x^2')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
```
![image](https://github.com/vacu9708/Machine-learning/assets/67142421/fa480064-b519-44f8-bfd7-065ec70c5f6e)
