# pyvectorflow: Python Bindings for the Vectorflow Neural Network Library

## Overview

`pyvectorflow` provides high-performance Python bindings for the `Vectorflow` library, a lightweight neural network framework written in D optimized for CPU-based training, especially suitable for sparse data. This library allows Python users to leverage the speed and efficiency of Vectorflow's D implementation directly within their Python applications.

## Why Ctypes?

The bindings are implemented using Python's `ctypes` foreign function interface. This approach was chosen for its robustness and direct interaction with the compiled D shared library. Earlier attempts with higher-level binding generators like `pyd` encountered persistent environment and compatibility issues, leading to segmentation faults. `ctypes` offers a simpler, more transparent, and dependency-free method, making the integration more stable and easier to debug.

## Installation

To install `pyvectorflow`, you need the Dlang toolchain (specifically `ldc2` and `dub`) installed and configured on your system.

### Prerequisites

1.  **Dlang Toolchain**: Install [LDC (LLVM D Compiler)](https://ldc.dlang.org/) and `dub` (the D package manager). A common way to do this is using the official Dlang installer or by manually setting up LDC. Ensure `ldc2` and `dub` are accessible in your PATH. If you used the Dlang installer, you likely sourced an `activate` script (e.g., `source ~/dlang/ldc-1.41.0/activate`).

    Alternatively, if `dacti` is an alias for your Dlang activation script (e.g., `source ~/dlang/ldc-1.41.0/activate`), ensure your shell environment is set up to use it.

2.  **Python Environment**: A Python 3.8+ environment. It's recommended to use a virtual environment (`venv` or `conda`).

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone git@github.com:apoorvalal/vectorflow.git
    cd vectorflow
    ```

2.  **Install Python Package**:
    Navigate to the `pyvectorflow` subdirectory and install it in editable mode. This command will also trigger the D shared library compilation.
    ```bash
    cd pyvectorflow
    # Activate your Python virtual environment first
    source /path/to/your/venv/bin/activate # or conda activate your_env
    pip install -e .
    ```

    The `pip install -e .` command will:
    *   Execute the `BuildDLibrary` custom command in `setup.py`.
    *   Compile the D shared library (`libvectorflow.so`) using `dub` and `ldc2`.
    *   Copy `libvectorflow.so` into the `pyvectorflow/vectorflow` Python package directory.
    *   Install the Python package in editable mode, allowing for live code changes.

## Usage

The `pyvectorflow` library exposes a `NeuralNet` class along with various layers and optimizers.

### Basic Neural Network Construction

```python
from vectorflow import NeuralNet, ADAM

# Create a neural network
nn = NeuralNet()

# Stack layers
nn.stack_DenseData(784) # Input layer for 784-dimensional dense data (e.g., 28x28 MNIST images)
nn.stack_Linear(200)    # Hidden layer with 200 neurons
nn.stack_ReLU()         # ReLU activation
nn.stack_DropOut(0.3)   # Dropout for regularization
nn.stack_Linear(10)     # Output layer with 10 neurons (e.g., 10 classes)
nn.stack_SeLU()         # Another activation example
nn.stack_TanH()         # Another activation example

# Initialize network parameters
nn.initialize(0.0001)

print(f"Network parameters: {nn.num_params}")
```

### Regularization

Regularizers can be added to the *last stacked `Linear` layer*.

```python
from vectorflow import NeuralNet, ADAM

nn_reg = NeuralNet()
nn_reg.stack_DenseData(100)
nn_reg.stack_Linear(50)
nn_reg.add_regularizer_L1(0.01) # Add L1 regularization to the last Linear layer
nn_reg.add_regularizer_L2(0.001) # Add L2 regularization
nn_reg.add_regularizer_Positive(0.0) # Ensure weights are positive
nn_reg.stack_ReLU()
nn_reg.stack_Linear(10)
nn_reg.initialize(0.01)
```

### Training

The `fit` method handles training with dense or sparse data.

```python
from vectorflow import NeuralNet, ADAM
import numpy as np

# Assuming X_train, y_train are NumPy arrays
X_train = np.random.rand(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.float32)

nn = NeuralNet()
nn.stack_DenseData(784).stack_Linear(10).initialize(0.01)

optimizer = ADAM(epochs=5, lr=0.001, batch_size=32)
nn.fit(X_train, y_train, optimizer=optimizer, verbose=True, num_cores=4)
```

**Sparse Data Training (using SciPy CSR matrix):**

```python
from vectorflow import NeuralNet, ADAM
import numpy as np
from scipy.sparse import csr_matrix

# Example sparse data (replace with actual data)
sparse_X_train = csr_matrix(np.random.rand(1000, 784) > 0.8, dtype=np.float32)
sparse_y_train = np.random.randint(0, 10, 1000).astype(np.float32)

nn_sparse = NeuralNet()
nn_sparse.stack_SparseData(784).stack_Linear(10).initialize(0.01)

optimizer_sparse = ADAM(epochs=5, lr=0.001, batch_size=32)
nn_sparse.fit(sparse_X_train, sparse_y_train, optimizer=optimizer_sparse, verbose=True, num_cores=4)
```

### Prediction

The `predict` method performs inference.

```python
# Assuming X_test_sample is a NumPy array for dense input
X_test_sample = np.random.rand(784).astype(np.float32)
predictions = nn.predict(X_test_sample, output_dim=10) # Specify output_dim for clarity
print(f"Predictions: {predictions}")
```

### API Reference (Key Classes)

*   `class NeuralNet`: Main network object.
    *   `__init__()`: Constructor.
    *   `stack_DenseData(dim)`: Adds a dense input layer.
    *   `stack_SparseData(dim)`: Adds a sparse input layer.
    *   `stack_Linear(dim)`: Adds a linear (dense) layer.
    *   `stack_ReLU()`: Adds a ReLU activation layer.
    *   `stack_SeLU()`: Adds a SeLU activation layer.
    *   `stack_TanH()`: Adds a TanH activation layer.
    *   `stack_DropOut(rate)`: Adds a Dropout layer.
    *   `stack_SparseKernelExpander(dim_out, cross_feats_str, max_group_id)`: Adds a sparse kernel expander layer.
    *   `add_regularizer_L1(lambda_val)`: Adds L1 regularization to the last linear layer.
    *   `add_regularizer_L2(lambda_val)`: Adds L2 regularization to the last linear layer.
    *   `add_regularizer_Positive(eps)`: Adds positive weight constraint to the last linear layer.
    *   `initialize(scale)`: Initializes network weights.
    *   `num_params` (property): Returns the total number of learnable parameters.
    *   `fit(X, y, optimizer, ...)`: Trains the network. Handles both dense (NumPy) and sparse (SciPy CSR) input.
    *   `predict(features, output_dim)`: Performs inference.
    *   `serialize(path)`: Saves the model to a file.
    *   `deserialize(path)`: Loads the model from a file.

*   `class Optimizer`: Base class for optimizers.
    *   `ADAM(...)`: ADAM optimizer.
    *   `AdaGrad(...)`: AdaGrad optimizer.

## Example: MNIST Classification (Full Script)

See `pyvectorflow/examples/test_package_mnist.py` for a complete example of training and evaluating a neural network on the MNIST dataset using `pyvectorflow`.

## License

`pyvectorflow` is licensed under the Apache-2.0 License.
