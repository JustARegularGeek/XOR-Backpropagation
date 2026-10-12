import numpy as np
import os

def load_parameters(filepath):
    """
    Reads parameters.txt and returns a dictionary of configuration values.
    Example output: {'learningRate': 0.3, 'maxIterations': 5000, ...}
    """
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
    """
    A Feedforward Neural Network implemented from scratch using NumPy,
    supporting customizable hidden layers, momentum-based gradient descent, and online learning.
    """

    def sigmoid(self, x):
        """Standard sigmoid activation function mapping any real value to the range (0, 1)."""
        return 1 / (1 + np.exp(-x))

    def __init__(self, params):
        """Initializes network architecture, weights, biases, and momentum buffers based on config parameters."""
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
        """
        Executes a forward pass through the network given an input pattern.
        Saves intermediate layer outputs for use during backpropagation.
        """
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
        """
        Computes local gradients (deltas) and updates all network weights 
        and biases using online gradient descent with momentum.
        """
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

    def evaluate_dataset(self, inputs, targets):
            """Evaluates overall Mean Squared Error and classification success rate on a given dataset."""
            total_error = 0.0
            correct_predictions = 0
            num_patterns = len(inputs)
    
            for i in range(num_patterns):
                output = self.forward_pass(inputs[i])
    
                # Error calculation for this pattern
                total_error += np.sum((targets[i] - output) ** 2) / 2
    
            # Check if rounded output matches target
            if np.all(np.round(output) == targets[i]):
                correct_predictions += 1
    
            mse = total_error / num_patterns
            success_rate = (correct_predictions / num_patterns) * 100
    
            return mse, success_rate    
        
    def train(self, training_inputs, expected_outputs, test_inputs, test_outputs, num_epochs):
        """
        Runs the online training loop over multiple epochs with random shuffling,
        tracks evaluation errors/success rates, and writes output files.
        """
        num_patterns = len(training_inputs)

        # Lists to store history for file generation
        error_history = []
        success_history = []

        for epoch in range(num_epochs):
            indices = np.arange(num_patterns)
            np.random.shuffle(indices)

            for i in indices: 
                current_input = training_inputs[i]
                target_output = expected_outputs[i]

                # Forward pass
                self.forward_pass(current_input)

                # Backward pass
                self.backward_pass(target_output)

            # Evaluate training set performance at end of epoch
            train_mse, train_success = self.evaluate_dataset(training_inputs, expected_outputs)
                
            # Evaluate testing set performance at end of epoch
            test_mse, test_success = self.evaluate_dataset(test_inputs, test_outputs)

            # Store for files: [epoch, train_error, test_error]
            error_history.append([epoch + 1, train_mse, test_mse])
            # Store for files: [epoch, train_success, test_success]
            success_history.append([epoch + 1, train_success, test_success])

        # Get current directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        errors_path = os.path.join(script_dir, "errors.txt")
        success_path = os.path.join(script_dir, "successrate.txt")

        # Save errors.txt (3 columns: epoch, train_error, test_error)
        np.savetxt(errors_path, error_history, fmt=["%d", "%.6f", "%.6f"])
        print(f"Saved errors to: {errors_path}")

        # Save successrate.txt (3 columns: epoch, train_success, test_success)
        np.savetxt(success_path, success_history, fmt=["%d", "%.2f", "%.2f"])
        print(f"Saved success rates to: {success_path}")

        # Print final patterns to screen
        print("--- Final Results on Patterns ---")
        print(f"{'Input':<20} | {'Desired (t)':<12} | {'Actual (o)':<12}")
        print("-" * 52)

        for i in range(num_patterns):
            actual = self.forward_pass(training_inputs[i])

            # Convert inputs and targets to integers for a cleaner look (assumes binary 0/1 inputs)
            clean_input = training_inputs[i].astype(int)
            clean_target = expected_outputs[i].astype(int)

            print(f"{str(clean_input):<20} | {str(clean_target):<12} | {str(np.round(actual, 4)):<12}")            

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

    # Load data
    test_data = np.loadtxt(params["testFile"])

    # Split the data into inputs and expected outputs
        # Grabs all rows, and columns from index 0 up to num_inputs
    test_inputs = test_data[:, :num_inputs]

    # Grabs all rows, and the final 'num_outputs' columns
    test_outputs = test_data[:, -num_outputs:]

    # Get maxIterations from parameters for the number of epochs
    epochs = int(params['maxIterations'])
    
    # Pass both arrays into your train function
    nn.train(training_inputs, expected_outputs, test_inputs, test_outputs, epochs)