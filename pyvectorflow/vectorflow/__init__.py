import ctypes
import numpy as np
import os
import glob

# Load library
# We try to find the library in the same directory as this file
_lib_path = os.path.join(os.path.dirname(__file__), "libvectorflow.so")
if not os.path.exists(_lib_path):
    # Fallback for development/testing if not installed yet
    _candidates = glob.glob(
        os.path.join(os.path.dirname(__file__), "../../libvectorflow.so")
    )
    if _candidates:
        _lib_path = _candidates[0]
    else:
        raise RuntimeError("Could not find libvectorflow.so")

lib = ctypes.CDLL(_lib_path)

# --- Define Argument and Return Types ---

# Runtime
lib.vf_init.restype = None
lib.vf_init.argtypes = []

# Lifecycle
lib.vf_NeuralNet_create.restype = ctypes.c_void_p
lib.vf_NeuralNet_create.argtypes = []

lib.vf_NeuralNet_destroy.restype = None
lib.vf_NeuralNet_destroy.argtypes = [ctypes.c_void_p]

# Layer Stacking
lib.vf_NeuralNet_stack_DenseData.restype = None
lib.vf_NeuralNet_stack_DenseData.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

lib.vf_NeuralNet_stack_SparseData.restype = None
lib.vf_NeuralNet_stack_SparseData.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

lib.vf_NeuralNet_stack_Linear.restype = None
lib.vf_NeuralNet_stack_Linear.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

lib.vf_NeuralNet_stack_ReLU.restype = None
lib.vf_NeuralNet_stack_ReLU.argtypes = [ctypes.c_void_p]

lib.vf_NeuralNet_stack_SeLU.restype = None
lib.vf_NeuralNet_stack_SeLU.argtypes = [ctypes.c_void_p]

lib.vf_NeuralNet_stack_TanH.restype = None
lib.vf_NeuralNet_stack_TanH.argtypes = [ctypes.c_void_p]

lib.vf_NeuralNet_stack_DropOut.restype = None
lib.vf_NeuralNet_stack_DropOut.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.vf_NeuralNet_stack_SparseKernelExpander.restype = None
lib.vf_NeuralNet_stack_SparseKernelExpander.argtypes = [
    ctypes.c_void_p,
    ctypes.c_ulong,
    ctypes.c_char_p,
    ctypes.c_uint,
]

# Regularizers
lib.vf_NeuralNet_add_regularizer_L1.restype = None
lib.vf_NeuralNet_add_regularizer_L1.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.vf_NeuralNet_add_regularizer_L2.restype = None
lib.vf_NeuralNet_add_regularizer_L2.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.vf_NeuralNet_add_regularizer_Positive.restype = None
lib.vf_NeuralNet_add_regularizer_Positive.argtypes = [ctypes.c_void_p, ctypes.c_float]

# Optimizers
lib.vf_Optimizer_create_ADAM.restype = ctypes.c_void_p
lib.vf_Optimizer_create_ADAM.argtypes = [
    ctypes.c_ulong,
    ctypes.c_float,
    ctypes.c_ulong,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
]

lib.vf_Optimizer_create_AdaGrad.restype = ctypes.c_void_p
lib.vf_Optimizer_create_AdaGrad.argtypes = [
    ctypes.c_ulong,
    ctypes.c_float,
    ctypes.c_ulong,
    ctypes.c_float,
]

lib.vf_Optimizer_destroy.restype = None
lib.vf_Optimizer_destroy.argtypes = [ctypes.c_void_p]

# Operations
lib.vf_NeuralNet_initialize.restype = None
lib.vf_NeuralNet_initialize.argtypes = [ctypes.c_void_p, ctypes.c_float]

lib.vf_NeuralNet_num_params.restype = ctypes.c_ulong
lib.vf_NeuralNet_num_params.argtypes = [ctypes.c_void_p]

lib.vf_NeuralNet_serialize.restype = None
lib.vf_NeuralNet_serialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

lib.vf_NeuralNet_deserialize.restype = None
lib.vf_NeuralNet_deserialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# Prediction
lib.vf_NeuralNet_predict.restype = None
lib.vf_NeuralNet_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ulong,
]

lib.vf_NeuralNet_predict_sparse.restype = None
lib.vf_NeuralNet_predict_sparse.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ulong,
]

# Training
lib.vf_NeuralNet_learn.restype = None
lib.vf_NeuralNet_learn.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_ulong,
    ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_char_p, # loss_type
    ctypes.c_bool,
    ctypes.c_uint,
]

lib.vf_NeuralNet_learn_sparse.restype = None
lib.vf_NeuralNet_learn_sparse.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # vals
    ctypes.POINTER(ctypes.c_uint),  # indices
    ctypes.POINTER(ctypes.c_ulong),  # indptr
    ctypes.c_ulong,  # n_samples
    ctypes.POINTER(ctypes.c_float),  # y
    ctypes.c_void_p,  # optimizer
    ctypes.c_char_p, # loss_type
    ctypes.c_bool,
    ctypes.c_uint,
]

# Initialize runtime once
lib.vf_init()


class Optimizer:
    def __init__(self, obj):
        self.obj = obj

    def __del__(self):
        if hasattr(self, "obj") and self.obj:
            lib.vf_Optimizer_destroy(self.obj)


class ADAM(Optimizer):
    def __init__(
        self, epochs=5, lr=0.01, batch_size=200, eps=1e-8, beta1=0.9, beta2=0.999
    ):
        obj = lib.vf_Optimizer_create_ADAM(
            ctypes.c_ulong(epochs),
            ctypes.c_float(lr),
            ctypes.c_ulong(batch_size),
            ctypes.c_float(eps),
            ctypes.c_float(beta1),
            ctypes.c_float(beta2),
        )
        super().__init__(obj)


class AdaGrad(Optimizer):
    def __init__(self, epochs=5, lr=0.01, batch_size=200, eps=1e-8):
        obj = lib.vf_Optimizer_create_AdaGrad(
            ctypes.c_ulong(epochs),
            ctypes.c_float(lr),
            ctypes.c_ulong(batch_size),
            ctypes.c_float(eps),
        )
        super().__init__(obj)


class NeuralNet:
    def __init__(self):
        self.obj = lib.vf_NeuralNet_create()
        self._initialized = False

    def __del__(self):
        if hasattr(self, "obj") and self.obj:
            lib.vf_NeuralNet_destroy(self.obj)

    def stack_DenseData(self, dim):
        lib.vf_NeuralNet_stack_DenseData(self.obj, dim)
        return self

    def stack_SparseData(self, dim):
        lib.vf_NeuralNet_stack_SparseData(self.obj, dim)
        return self

    def stack_Linear(self, dim):
        lib.vf_NeuralNet_stack_Linear(self.obj, dim)
        return self

    def stack_ReLU(self):
        lib.vf_NeuralNet_stack_ReLU(self.obj)
        return self

    def stack_SeLU(self):
        lib.vf_NeuralNet_stack_SeLU(self.obj)
        return self

    def stack_TanH(self):
        lib.vf_NeuralNet_stack_TanH(self.obj)
        return self

    def stack_DropOut(self, rate):
        lib.vf_NeuralNet_stack_DropOut(self.obj, rate)
        return self

    def stack_SparseKernelExpander(self, dim_out, cross_feats_str, max_group_id=100):
        c_str = ctypes.c_char_p(cross_feats_str.encode("utf-8"))
        lib.vf_NeuralNet_stack_SparseKernelExpander(
            self.obj, dim_out, c_str, max_group_id
        )
        return self

    def add_regularizer_L1(self, lam):
        lib.vf_NeuralNet_add_regularizer_L1(self.obj, lam)
        return self

    def add_regularizer_L2(self, lam):
        lib.vf_NeuralNet_add_regularizer_L2(self.obj, lam)
        return self

    def add_regularizer_Positive(self, eps=0.0):
        lib.vf_NeuralNet_add_regularizer_Positive(self.obj, eps)
        return self

    def initialize(self, scale=0.01):
        lib.vf_NeuralNet_initialize(self.obj, ctypes.c_float(scale))
        self._initialized = True

    def serialize(self, path):
        c_path = ctypes.c_char_p(path.encode("utf-8"))
        lib.vf_NeuralNet_serialize(self.obj, c_path)

    def deserialize(self, path):
        c_path = ctypes.c_char_p(path.encode("utf-8"))
        lib.vf_NeuralNet_deserialize(self.obj, c_path)
        self._initialized = True

    @property
    def num_params(self):
        return lib.vf_NeuralNet_num_params(self.obj)

    def predict(self, features, output_dim=None):
        # Infer if dense or sparse
        # For now, simple implementation: assume dense numpy array

        # TODO: Handle scipy.sparse.csr_matrix for sparse prediction

        if hasattr(features, "format") and features.format == "csr":
            # Sparse case
            raise NotImplementedError(
                "Single sparse prediction wrapper not yet implemented. Use predict_sparse for raw arrays."
            )

        features = np.ascontiguousarray(features, dtype=np.float32)
        n_features = len(features)

        # Heuristic for output buffer size if not provided
        # We should ideally query the net for output dim.
        if output_dim is None:
            output_dim = 100  # Safe default?

        out_buf = np.zeros(output_dim, dtype=np.float32)

        lib.vf_NeuralNet_predict(
            self.obj,
            features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_ulong(n_features),
            out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_ulong(len(out_buf)),
        )

        return out_buf

    def fit(
        self,
        X,
        y,
        optimizer=None,
        loss="multinomial",
        epochs=5,
        lr=0.01,
        batch_size=200,
        verbose=True,
        num_cores=1,
    ):
        if optimizer is None:
            optimizer = ADAM(epochs, lr, batch_size)

        y = np.ascontiguousarray(y, dtype=np.float32)
        n_samples = X.shape[0]
        loss_cstr = ctypes.c_char_p(loss.encode("utf-8"))

        # Check if X is sparse (scipy.sparse)
        if hasattr(X, "format") and X.format == "csr":
            # Handle Sparse Input
            vals = X.data.astype(np.float32)
            indices = X.indices.astype(np.uint32)
            indptr = X.indptr.astype(np.uint64)  # Assuming 64-bit offsets

            lib.vf_NeuralNet_learn_sparse(
                self.obj,
                vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),
                ctypes.c_ulong(n_samples),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                optimizer.obj,
                loss_cstr,
                ctypes.c_bool(verbose),
                ctypes.c_uint(num_cores),
            )
        else:
            # Dense Input
            X = np.ascontiguousarray(X, dtype=np.float32)
            n_features = X.shape[1]

            lib.vf_NeuralNet_learn(
                self.obj,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_ulong(n_samples),
                ctypes.c_ulong(n_features),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                optimizer.obj,
                loss_cstr,
                ctypes.c_bool(verbose),
                ctypes.c_uint(num_cores),
            )
