#!/usr/bin/env python
import numpy as np
import time
from vectorflow import NeuralNet, ADAM

from mnist import load_mnist_data


def main():
    print("MNIST with PyVectorflow (Installed Package)")

    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist_data()

    # Create network
    nn = NeuralNet()
    nn.stack_DenseData(28 * 28)
    nn.stack_Linear(200)
    nn.stack_DropOut(0.3)
    nn.stack_SeLU()
    nn.stack_Linear(10)

    nn.initialize(0.0001)

    print(f"Network parameters: {nn.num_params}")

    start = time.time()
    # Using explicit optimizer
    optimizer = ADAM(epochs=5, lr=0.001, batch_size=200)
    nn.fit(X_train, y_train, optimizer=optimizer, verbose=True, num_cores=4)
    duration = time.time() - start

    print(f"\nTraining time: {duration:.2f} seconds")

    # Test
    correct = 0
    # Test on a subset to be quick
    n_test = len(X_test)
    print(f"Testing on {n_test} samples...")

    # Batch prediction logic could be added to wrapper, but single predict loop for now
    start_test = time.time()
    for i in range(n_test):
        pred = nn.predict(X_test[i])
        if np.argmax(pred) == y_test[i]:
            correct += 1

    test_duration = time.time() - start_test
    print(f"Testing time: {test_duration:.2f} seconds")

    accuracy = correct / n_test
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
