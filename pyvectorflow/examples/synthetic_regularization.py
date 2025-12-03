#!/usr/bin/env python
import numpy as np
from vectorflow import NeuralNet, ADAM, AdaGrad
import time

def generate_synthetic_data(n_samples=5000, n_features=20, n_informative=5, random_state=42):
    """
    Generates synthetic binary classification data.
    Data is generated from a linear model: y = 1 if (X * w + b) > 0 else 0
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate weights for a ground truth linear model
    # Sparse weights: only first n_informative are non-zero
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = 3.0 * np.random.randn(n_informative) # Strong signal

    # Generate labels
    logits = np.dot(X, true_weights)
    # Add some noise to make it not perfectly separable if desired, but let's keep it clean for sanity first
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(np.float32)

    # Vectorflow expects labels to be 1.0 or -1.0 for logistic loss internally,
    # but the binding converts >0 to 1 and <=0 to -1 automatically if we use the standard loss function.
    # However, to be explicit and match the RCV1 example style logic:
    # 0 -> -1, 1 -> 1.
    # But our C wrapper just takes float. The internal D code does `label > 0 ? 1.0 : -1.0`.
    # So passing 0.0 and 1.0 is perfectly fine.

    return X, y

def train_model(name, regularization_type=None, lambda_val=0.0):
    print(f"\n--- Training Model: {name} ---")

    # 1. Prepare Data
    n_features = 50
    X_train, y_train = generate_synthetic_data(n_samples=10000, n_features=n_features, n_informative=10)
    X_test, y_test = generate_synthetic_data(n_samples=2000, n_features=n_features, n_informative=10, random_state=43)

    print(f"Train data shape: {X_train.shape}")

    # 2. Build Network (Linear Model)
    # Architecture: Input -> Linear(1) -> Output
    # This is equivalent to Logistic Regression when trained with logistic loss.
    nn = NeuralNet()

    input_dim = X_train.shape[1]

    nn.stack_DenseData(input_dim)
    nn.stack_Linear(1) # Linear decision boundary

    # Apply Regularization to the Linear layer
    if regularization_type == 'L1':
        print(f"Adding L1 Regularization (Lasso) with lambda={lambda_val}")
        nn.add_regularizer_L1(lambda_val)
    elif regularization_type == 'L2':
        print(f"Adding L2 Regularization (Ridge) with lambda={lambda_val}")
        nn.add_regularizer_L2(lambda_val)

    nn.initialize(0.0001) # Initialize weights to 0 for convex problem

    # 3. Train
    # AdaGrad is often faster for linear/convex problems
    optimizer = AdaGrad(epochs=100, lr=0.001, batch_size=200)

    # Debug: Check labels distribution
    print(f"y_train distribution: {np.mean(y_train)} (should be approx 0.5)")

    start = time.time()
    nn.fit(X_train, y_train, optimizer=optimizer, loss='logistic', verbose=True, num_cores=8)
    print(f"Training took {time.time() - start:.2f}s")

    # 4. Evaluate
    # Loop over test set for prediction (since current Python wrapper predicts one sample at a time)
    preds = []
    for i in range(X_test.shape[0]):
        p = nn.predict(X_test[i], output_dim=1)
        preds.append(p[0])

    preds = np.array(preds)

    # Predictions > 0 are class 1, < 0 are class 0 (mapped to -1 internally)
    pred_labels = (preds > 0).astype(np.float32)

    accuracy = np.mean(pred_labels == y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    # Baseline
    train_model("Baseline (No Regularization)")

    # Lasso (L1)
    train_model("Lasso Regression (L1)", 'L1', 0.01)

    # Ridge (L2)
    train_model("Ridge Regression (L2)", 'L2', 0.01)
