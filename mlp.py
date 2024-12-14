import numpy as np

# Define the sigmoid activation function and its derivative
def activation_function(x):
    return 1 / (1 + np.exp(-x))

def activation_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, layer_dimensions):
        self.synaptic_weights = [np.random.uniform(-1, 1, (layer_dimensions[i], layer_dimensions[i + 1])) for i in range(len(layer_dimensions) - 1)]
        self.bias_terms = [np.random.uniform(-1, 1, (1, layer_dimensions[i + 1])) for i in range(len(layer_dimensions) - 1)]

    def propagate_forward(self, input_data):
        self.layer_pre_activations = []  # Store pre-activation values for each layer
        self.layer_activations = []  # Store post-activation outputs for each layer

        current_layer_output = np.array(input_data)
        for weight_matrix, bias_vector in zip(self.synaptic_weights, self.bias_terms):
            self.layer_pre_activations.append(current_layer_output)
            z = np.dot(current_layer_output, weight_matrix) + bias_vector
            current_layer_output = activation_function(z)
            self.layer_activations.append(current_layer_output)

        return current_layer_output

    def propagate_backward(self, target_output, learning_rate):
        # Compute the error at the output layer
        output_error = self.layer_activations[-1] - target_output
        for layer_index in reversed(range(len(self.synaptic_weights))):
            layer_output = self.layer_activations[layer_index]
            input_to_layer = self.layer_pre_activations[layer_index]

            # Calculate gradients
            delta = output_error * activation_derivative(layer_output)
            weight_adjustments = np.dot(input_to_layer.T, delta)
            bias_adjustments = np.sum(delta, axis=0, keepdims=True)

            # Update weights and biases
            self.synaptic_weights[layer_index] -= learning_rate * weight_adjustments
            self.bias_terms[layer_index] -= learning_rate * bias_adjustments

            # Backpropagation the error to the previous layer
            output_error = np.dot(delta, self.synaptic_weights[layer_index].T)

    def train_network(self, training_data, training_targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            cumulative_loss = 0
            for input_sample, target_sample in zip(training_data, training_targets):
                input_sample = np.array(input_sample).reshape(1, -1)
                target_sample = np.array(target_sample).reshape(1, -1)
                predictions = self.propagate_forward(input_sample)
                loss = np.mean((predictions - target_sample) ** 2)
                cumulative_loss += loss
                self.propagate_backward(target_sample, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {cumulative_loss:.4f}")

    def make_prediction(self, input_data):
        input_data = np.array(input_data).reshape(1, -1)
        return self.propagate_forward(input_data)

# XOR Problem Dataset
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])

# Initialize and train the Neural Network
neural_net = NeuralNetwork([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output
neural_net.train_network(xor_inputs, xor_outputs, num_epochs=10000, learning_rate=0.1)

# Test the Neural Network
print("Testing the trained MLP:")
for sample_input in xor_inputs:
    prediction = neural_net.make_prediction(sample_input)
    print(f"Input: {sample_input}, Output: {prediction.flatten()}")