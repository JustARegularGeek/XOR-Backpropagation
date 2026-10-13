# XOR Neural Network (Python / NumPy)

A feed-forward neural network implemented **from scratch** with NumPy and trained with **back propagation (on-line update, with momentum)** to learn the XOR function, a problem that is not linearly separable and cannot be solved by a single-layer perceptron.

---

## Features

- **Flexible architecture:** one or two hidden layers. If `numHiddenLayerTwoNeurons` is `0`, the network has a single hidden layer.
- **Sigmoid activation** in all hidden and output neurons, and a **trainable bias** for every non-input neuron.
- **Back propagation with momentum:** `Δw(t) = η·δ·input + μ·Δw(t−1)` with learning rate η and momentum factor μ.
- **On-line learning:** weights are updated after every pattern; patterns are shuffled at the start of every epoch.
- **Reproducible:** weights and biases are drawn uniformly from [−1, 1] using a fixed seed (`randomSeed`).
- **External configuration:** all hyperparameters and file names are read from `parameters.txt` (nothing is hard-coded).
- **Metrics per epoch** for both the training and the test set:
  - error `E = (1/P)·Σ(t − o)²` (mean squared error over the P patterns),
  - success rate (%), where an output ≥ 0.5 counts as 1 and otherwise 0.
- Training runs for a fixed number of epochs (`maxIterations`); there is no early stopping.

---

## Project structure

```text
XOR-Backpropagation/
├── code/
│   ├── main.py            # NeuralNetwork class and training script
│   ├── plot_results.py    # draws the error and success-rate plots
│   ├── parameters.txt     # hyperparameters and settings
│   ├── training.txt       # training patterns
│   ├── test.txt           # test patterns
│   ├── errors.txt         # (generated) epoch, training error, testing error
│   ├── successrate.txt    # (generated) epoch, training %, testing %
│   ├── error_plot.png     # (generated) error plot
│   └── success_plot.png   # (generated) success-rate plot
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.8 or newer
- `numpy` (network and file handling)
- `matplotlib` (plots only)

No machine-learning libraries (PyTorch, TensorFlow, Keras, scikit-learn) are used.

---

## Environment setup

1. Clone the repository:

   ```bash
   git clone https://github.com/JustARegularGeek/XOR-Backpropagation.git
   cd XOR-Backpropagation
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv env
   source env/bin/activate        # Windows: env\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## How to run

The program reads `parameters.txt` from the **current folder**, so always run it from inside `code/`:

```bash
cd code
python main.py            # trains the network, writes errors.txt and successrate.txt
python plot_results.py    # (optional) creates error_plot.png and success_plot.png
```

---

## Configuration (`parameters.txt`)

One `name value` pair per line. Example (the main experiment):

```text
numHiddenLayerOneNeurons 4
numHiddenLayerTwoNeurons 4
numInputNeurons 2
numOutputNeurons 1
learningRate 0.5
momentum 0.9
maxIterations 5000
randomSeed 42
trainFile training.txt
testFile test.txt
```

| Parameter | Meaning |
|---|---|
| `numInputNeurons` | number of input columns (2 for XOR) |
| `numHiddenLayerOneNeurons` | neurons in the first hidden layer |
| `numHiddenLayerTwoNeurons` | neurons in the second hidden layer (`0` = no second hidden layer) |
| `numOutputNeurons` | number of output neurons (1 for XOR) |
| `learningRate` | learning rate η |
| `momentum` | momentum factor μ |
| `maxIterations` | number of epochs (one epoch = one pass over all training patterns) |
| `randomSeed` | seed for weight initialisation and pattern shuffling |
| `trainFile`, `testFile` | names of the data files |

To run another experiment, change the values in `parameters.txt` and run `python main.py` again.

---

## Data format

`training.txt` and `test.txt` have one pattern per line: the inputs followed by the desired output, separated by spaces. For XOR both files contain the same four patterns:

```text
0 0 0
0 1 1
1 0 1
1 1 0
```

---

## Output

- **`errors.txt`**: three columns per epoch: epoch, training error, testing error.
- **`successrate.txt`**: three columns per epoch: epoch, training success rate (%), testing success rate (%).
- **Screen**: at the end of training, the four patterns with the desired and the actual output. With the parameters above:

  ```text
  Input                | Desired (t)  | Actual (o)
  ----------------------------------------------------
  [0 0]                | [0]          | [0.003]
  [0 1]                | [1]          | [0.9966]
  [1 0]                | [1]          | [0.9968]
  [1 1]                | [0]          | [0.0047]
  ```

- **`error_plot.png`, `success_plot.png`** (from `plot_results.py`): training and testing error and success rate per epoch.

With the parameters above the network reaches 100% success from about epoch 200 and the error at epoch 5000 is about 1.3×10⁻⁵.