import numpy as np
import matplotlib.pyplot as plt

# Summation Unit
def summation_unit(inputs, weights):
    return np.dot(inputs, weights)

# Activation Functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def tanh_function(x):
    return np.tanh(x)

def relu_function(x):
    return max(0, x)

def leaky_relu_function(x, alpha=0.01):
    return x if x > 0 else alpha * x

# Comparator for Error Calculation
def error_calculation(targets, outputs):
    return np.sum((targets - outputs) ** 2) / len(targets)

# Perceptron Training Function
def perceptron_train(inputs, targets, weights, learning_rate, activation_function, max_epochs=1000, convergence_error=0.002):
    epochs = 0
    errors = []
    
    while epochs < max_epochs:
        outputs = []
        total_error = 0
        
        for i in range(len(inputs)):
            summation = summation_unit(inputs[i], weights)
            output = activation_function(summation)
            outputs.append(output)
            error = targets[i] - output
            
            # Update weights
            weights += learning_rate * error * inputs[i]
            total_error += error ** 2
        
        errors.append(total_error)
        if total_error <= convergence_error:
            break
        epochs += 1

    return weights, epochs, errors

# Plotting Function
def plot_error(errors):
    if len(errors) == 0:
        print("No error data to plot.")
        return
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error')
    plt.grid()
    plt.show()

# A2: AND Gate Implementation with Step Activation
inputs = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

targets = np.array([0, 0, 0, 1])
weights = np.random.uniform(-0.5, 0.5, size=3)
learning_rate = 0.05

final_weights, epochs, errors = perceptron_train(inputs, targets, weights.copy(), learning_rate, step_function)
print("Final Weights:", final_weights)
print("Epochs Taken to Converge:", epochs)
plot_error(errors)

# A3: Comparison using Different Activation Functions
activation_functions = [bipolar_step_function, sigmoid_function, relu_function]
activation_names = ['Bipolar Step', 'Sigmoid', 'ReLU']

for func, name in zip(activation_functions, activation_names):
    _, epochs, _ = perceptron_train(inputs, targets, weights.copy(), learning_rate, func)
    print(f"{name} Activation - Epochs: {epochs}")

# A4: Varying Learning Rate Experiment
learning_rates = np.arange(0.1, 1.1, 0.1)
epochs_list = []

for lr in learning_rates:
    _, epochs, _ = perceptron_train(inputs, targets, weights.copy(), lr, step_function)
    epochs_list.append(epochs)

plt.plot(learning_rates, epochs_list, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Epochs to Converge')
plt.title('Learning Rate vs Epochs')
plt.grid()
plt.show()
