import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for value in random_values:
    sigmoid_value = sigmoid(value)
    print(f"Sigmoid of {value}: {sigmoid_value:.4f}")

for value in random_values:
    tanh_value = tanh(value)
    print(f"TanH of {value}: {tanh_value:.4f}")

for value in random_values:
    relu_value = relu(value)
    print(f"ReLu of {value}: {relu_value:.4f}")

for value in random_values:
    leaky_relu_value = leaky_relu(value)
    print(f"Leaky ReLu of {value}: {leaky_relu_value:.4f}")