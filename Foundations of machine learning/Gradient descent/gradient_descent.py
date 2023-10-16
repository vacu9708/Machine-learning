def function(x):
    return x**2

def get_gradient(x):
    return 2*x

def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    history = []  # to store the path of x during descent

    for i in range(num_iterations):
        grad = get_gradient(x)
        x = x - learning_rate * grad
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
