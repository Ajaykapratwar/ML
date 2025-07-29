import numpy as np
import matplotlib.pyplot as plt

# Sample Data (x: features, y: target values)
X = np.array([1, 2, 3, 4, 5])  # Input feature
y = np.array([2, 3, 4, 5, 6])  # Target values

# Reshape X for matrix operations
X = X.reshape(-1, 1)
m = len(y)  # Number of examples

# Add bias term (column of ones)
X_b = np.c_[np.ones((m, 1)), X]  # Now X_b = [1, x] for each row

# Initialize parameters
theta = np.zeros(2)  # [theta_0, theta_1]
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Train the model
theta_final, cost_history = gradient_descent(X_b, y, theta, alpha, iterations)

# Print final parameters
print("Final parameters:", theta_final)

# Plot Cost Function over iterations
plt.plot(range(iterations), cost_history, label="Cost Function J(θ)")
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Cost Function Reduction Over Time")
plt.legend()
plt.show()