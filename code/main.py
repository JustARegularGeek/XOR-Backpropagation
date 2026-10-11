import numpy as np

def load_parameters(filepath):
    params = {}

    with open("parameters.txt") as file:

        for line in file:
            temp = line.split()

            # Check if the line actually has two parts to prevent errors on empty lines
            if len(temp) == 2:
                key = temp[0]
                value = temp[1]
                # Assign the value to the dictionary using the key
                params[key] = value

    return params

class NeuralNetwork:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, params):
        # Cast parameters to the correct data types
        self.learning_rate = float(params['learningRate'])
        self.momentum = float(params['momentum'])
        self.input_size = int(params['numInputNeurons'])
        self.hidden_layer1_size = int(params['numHiddenLayerOneNeurons'])
        self.hidden_layer2_size = int(params['numHiddenLayerTwoNeurons'])
        self.output_size = int(params['numOutputNeurons'])

        # Set random seed
        np.random.seed(int(params['randomSeed']))

        # --- Weight and bias arrays ---

        # Create and randomly initialize weights array between input layer and hidden layer 1
        self.weights_i_h1 = np.random.uniform(-1, 1, size = (self.input_size, self.hidden_layer1_size))

        # Create and randomly initialize bias array for hidden layer 1
        self.bias_h1 = np.random.uniform(-1, 1, size = (self.hidden_layer1_size,))

        # Check if hidden layer 2 exists
        if self.hidden_layer2_size > 0:

            # Create and randomly initialize weights array between hidden layer 1 and 2
            self.weights_h1_h2 = np.random.uniform(-1, 1, size = (self.hidden_layer1_size, self.hidden_layer2_size))

            # Create and randomly initialize bias array for hidden layer 2
            self.bias_h2 = np.random.uniform(-1, 1, size = (self.hidden_layer2_size,))

            # Create and randomly initialize weights array between hidden layer 2 and output layer
            self.weights_h2_o = np.random.uniform(-1, 1, size = (self.hidden_layer2_size, self.output_size))

        else:
            # Create and randomly initialize weights array between hidden layer 1 and output layer
            self.weights_h1_o = np.random.uniform(-1, 1, size = (self.hidden_layer1_size, self.output_size))

        # Create and randomly initialize bias array for output layer
        self.bias_o = np.random.uniform(-1, 1, size = (self.output_size,))


        # --- Momentum weight and bias arrays ---

        # Create and initialize momentum array between input layer and hidden layer 1
        self.momentum_i_h1 = np.zeros_like(self.weights_i_h1)

        # Create and initialize momentum bias array for layer 1
        self.momentum_bias_h1 = np.zeros_like(self.bias_h1)

        # Check if hidden layer 2 exists
        if self.hidden_layer2_size > 0:

            # Create and initialize momentum array between hidden layer 1 and hidden layer 2
            self.momentum_h1_h2 = np.zeros_like(self.weights_h1_h2)

            # Create and initialize momentum bias array for layer 2
            self.momentum_bias_h2 = np.zeros_like(self.bias_h2)

            # Create and initialize momentum array between hidden layer 2 and output layer
            self.momentum_h2_o = np.zeros_like(self.weights_h2_o)

        else:
            # Create and initialize momentum array between hidden layer 1 and output layer
            self.momentum_h1_o = np.zeros_like(self.weights_h1_o)

        # Create and initialize momentum bias array for output layer
        self.momentum_bias_o = np.zeros_like(self.bias_o)
                    
    def forward_pass(self, inputs):
        self.inputs = inputs

        # Calculate output for layer 1
        y_i_h1 = np.dot(inputs, self.weights_i_h1) + self.bias_h1
        self.out_h1 = self.sigmoid(y_i_h1)

        # Check if hidden layer 2 exists
        if self.hidden_layer2_size > 0:
            # Calculate output for layer 2
            y_h1_h2 = np.dot(self.out_h1, self.weights_h1_h2) + self.bias_h2
            self.out_h2 = self.sigmoid(y_h1_h2)

            # Calculate total output
            y_o = np.dot(self.out_h2, self.weights_h2_o) + self.bias_o

        else:
            # Calculate total output
            y_o = np.dot(self.out_h1, self.weights_h1_o) + self.bias_o

        # Calculate final activation output
        self.out = self.sigmoid(y_o)

        return self.out
        
    def backward_pass(self, expected_output):
        # Calculate output layer delta
        delta_output = self.out * (1 - self.out) * (expected_output - self.out)

        # Check if hidden layer 2 exists
        if self.hidden_layer2_size > 0:
            # Calculate hidden layer 2 delta
            delta_h2 = self.out_h2 * (1 - self.out_h2) * np.dot(delta_output, self.weights_h2_o.T)

            # Calculate hidden layer 1 delta
            delta_h1 = self.out_h1 * (1 - self.out_h1) * np.dot(delta_h2, self.weights_h1_h2.T)

        else:
            # Calculate hidden layer 1 delta
            delta_h1 = self.out_h1 * (1 - self.out_h1) * np.dot(delta_output, self.weights_h1_o.T)

        # Check if hidden layer 2 exists
        if self.hidden_layer2_size > 0:

            # Calculate weight and bias difference between layer 2 and output layer
            dWeight_h2_o = self.learning_rate * np.outer(self.out_h2, delta_output) + self.momentum * self.momentum_h2_o
            dBias_h2_o = self.learning_rate * delta_output + self.momentum * self.momentum_bias_o

            # wij(t + 1) = wij(t) + Δwij + μ[wij(t) - wij(t - 1)]
            self.weights_h2_o += dWeight_h2_o

            # biasj(t + 1) = biasj(t) + Δbiasj + μ[biasj(t) - biasj(t - 1)]
            self.bias_o += dBias_h2_o

            # Update weight and bias momentum 
            self.momentum_h2_o = dWeight_h2_o
            self.momentum_bias_o = dBias_h2_o

            # Calculate weight and bias difference between layer 1 and layer 2
            dWeight_h1_h2 = self.learning_rate * np.outer(self.out_h1, delta_h2) + self.momentum * self.momentum_h1_h2
            dBias_h2 = self.learning_rate * delta_h2 + self.momentum * self.momentum_bias_h2

            # wij(t + 1) = wij(t) + Δwij + μ[wij(t) - wij(t - 1)]
            self.weights_h1_h2 += dWeight_h1_h2

            # biasj(t + 1) = biasj(t) + Δbiasj + μ[biasj(t) - biasj(t - 1)]
            self.bias_h2 += dBias_h2

            # Update weight and bias momentum
            self.momentum_h1_h2 = dWeight_h1_h2
            self.momentum_bias_h2 = dBias_h2

        else:

            # Calculate weight and bias difference between layer 1 and output layer
            dWeight_h1_o = self.learning_rate * np.outer(self.out_h1, delta_output) + self.momentum * self.momentum_h1_o
            dBias_h1 = self.learning_rate * delta_h1 + self.momentum * self.momentum_bias_h1

            # wij(t + 1) = wij(t) + Δwij + μ[wij(t) - wij(t - 1)]
            self.weights_h1_o += dWeight_h1_o

            # biasj(t + 1) = biasj(t) + Δbiasj + μ[biasj(t) - biasj(t - 1)]
            self.bias_h1 += dBias_h1

            # Update weight and bias momentum
            self.momentum_h1_o = dWeight_h1_o
            self.momentum_bias_h1 = dBias_h1

        # Calculate weight and bias difference between input layer and layer 1 
        dW_i_h1 = self.learning_rate * np.outer(self.inputs, delta_h1) + self.momentum * self.momentum_i_h1
        dB_h1 = self.learning_rate * delta_h1 + self.momentum * self.momentum_bias_h1

        # wij(t + 1) = wij(t) + Δwij + μ[wij(t) - wij(t - 1)]
        self.weights_i_h1 += dW_i_h1

        # biasj(t + 1) = biasj(t) + Δbiasj + μ[biasj(t) - biasj(t - 1)]
        self.bias_h1 += dB_h1

        # Update weight and bias momentum
        self.momentum_i_h1 = dW_i_h1          
        self.momentum_bias_h1 = dB_h1
            
        
    def train(self, training_inputs, expected_outputs, num_epochs):
        # TODO: Loop for maxIterations (epochs). 
        # Inside the loop, shuffle data, call forward_pass and backward_pass for each row.

        num_patterns = len(training_inputs)

        for epoch in range(num_epochs):
            indices = np.arange(num_patterns)
            np.random.shuffle(indices)

            epoch_error = 0.0

            for i in indices: 
                current_input = training_inputs[i]
                target_output = expected_outputs[i]

                # Forward pass
                actual_output = self.forward_pass(current_input)

                # Backward pass
                self.backward_pass(target_output)

                # Calculate error
                pattern_error = np.sum(target_output - actual_output) ** 2 / 2

                # Add the error to the total epoch error
                epoch_error += pattern_error

            # Calculate mean square error
            mse = epoch_error / num_patterns

            # Print progress every 100 epochs so you can watch it learn
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | MSE: {mse}")

if __name__ == "__main__":
    # Load parameters
    params = load_parameters("parameters.txt")
    
    # Load data
    training_data = np.loadtxt(params["trainFile"])

    num_inputs = int(params['numInputNeurons'])
    num_outputs = int(params['numOutputNeurons'])

    # Split the data into inputs and expected outputs
    # Grabs all rows, and columns from index 0 up to num_inputs
    training_inputs = training_data[:, :num_inputs] 
    
    # Grabs all rows, and the final 'num_outputs' columns
    expected_outputs = training_data[:, -num_outputs:]
    
    # Initialize and train network
    nn = NeuralNetwork(params)
    
    # Get maxIterations from parameters for the number of epochs
    epochs = int(params['maxIterations'])
    
    # Pass both arrays into your train function
    nn.train(training_inputs, expected_outputs, epochs)