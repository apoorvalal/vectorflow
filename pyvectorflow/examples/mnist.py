import gzip
import os
import struct
import urllib.request
import numpy as np

def load_mnist_data(data_dir="mnist_data/"):
    """Load MNIST data from IDX format files"""
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if raw IDX files exist
    files_exist = all(
        os.path.exists(os.path.join(data_dir, f))
        for f in ["train", "train_labels", "test", "test_labels"]
    )

    if not files_exist:
        root_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

        # Download files if they don't exist
        files = [
            ("train-images-idx3-ubyte.gz", "train.gz"),
            ("train-labels-idx1-ubyte.gz", "train_labels.gz"),
            ("t10k-images-idx3-ubyte.gz", "test.gz"),
            ("t10k-labels-idx1-ubyte.gz", "test_labels.gz"),
        ]

        print("Downloading MNIST dataset...")
        for remote_file, local_file in files:
            print(f"Downloading {remote_file}...")
            urllib.request.urlretrieve(
                root_url + remote_file, os.path.join(data_dir, local_file)
            )

            # Decompress gz files
            with gzip.open(os.path.join(data_dir, local_file), "rb") as f_in:
                with open(os.path.join(data_dir, local_file[:-3]), "wb") as f_out:
                    f_out.write(f_in.read())

            # Remove gz files
            os.remove(os.path.join(data_dir, local_file))

    def read_idx_file(prefix):
        """Read IDX file format"""
        with (
            open(os.path.join(data_dir, prefix), "rb") as fx,
            open(os.path.join(data_dir, prefix + "_labels"), "rb") as fl,
        ):
            # Read magic number and assert it's correct (2051 for images)
            magic = struct.unpack(">i", fx.read(4))[0]
            assert magic == 2051, f"Wrong MNIST magic number. Corrupted data: {magic}"

            # Skip dimensions from image file
            fx.read(12)  # Skip num_images, num_rows, num_cols

            # Skip magic number and dimensions from label file
            fl.read(8)  # Skip magic number and num_items

            # Determine dataset size
            n = 60000 if prefix == "train" else 10000

            # Read data
            features = []
            labels = []
            for _ in range(n):
                label = struct.unpack("B", fl.read(1))[0]
                pixels = struct.unpack("B" * (28 * 28), fx.read(28 * 28))

                features.append(list(map(float, pixels)))
                labels.append(float(label))

            return np.array(features), np.array(labels)

    # Load train and test data
    train_features, train_labels = read_idx_file("train")
    test_features, test_labels = read_idx_file("test")

    return (train_features, train_labels), (test_features, test_labels)

if __name__ == "__main__":
    print("MNIST Data Loader (Standalone)")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print(f"Train data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
