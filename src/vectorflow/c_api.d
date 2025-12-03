module vectorflow.c_api;

import core.runtime;
import core.memory;
import std.string;
import std.stdio;

import vectorflow.neuralnet;
import vectorflow.layers;
import vectorflow.neurallayer;
import vectorflow.optimizers;
import vectorflow.regularizers;
import vectorflow.serde;

// Distinct structs for Dense and Sparse to avoid ambiguity
struct DenseObservation {
    float[] features;
    float label;
}

struct SparseObservation {
    SparseF[] features;
    float label;
}

extern(C) {

    // --- Runtime Management ---

    void vf_init() {
        try {
            Runtime.initialize();
        } catch (Exception e) {
            printf("Error initializing runtime: %.*s\n", cast(int)e.msg.length, e.msg.ptr);
        }
    }

    // --- NeuralNet Lifecycle ---

    void* vf_NeuralNet_create() {
        try {
            auto nn = new NeuralNet();
            GC.addRoot(cast(void*)nn);
            return cast(void*)nn;
        } catch (Throwable t) {
             printf("Error in create: %.*s\n", cast(int)t.msg.length, t.msg.ptr);
             return null;
        }
    }

    void vf_NeuralNet_destroy(void* ptr) {
        if (ptr) {
            GC.removeRoot(ptr);
        }
    }

    // --- Layer Stacking ---

    void vf_NeuralNet_stack_DenseData(void* ptr, ulong dim) {
        (cast(NeuralNet)ptr).stack(new DenseData(dim));
    }

    void vf_NeuralNet_stack_SparseData(void* ptr, ulong dim) {
        (cast(NeuralNet)ptr).stack(new SparseData(dim));
    }

    void vf_NeuralNet_stack_Linear(void* ptr, ulong dim) {
        (cast(NeuralNet)ptr).stack(new Linear(dim));
    }

    void vf_NeuralNet_stack_ReLU(void* ptr) {
        (cast(NeuralNet)ptr).stack(new ReLU());
    }

    void vf_NeuralNet_stack_SeLU(void* ptr) {
        (cast(NeuralNet)ptr).stack(new SeLU());
    }

    void vf_NeuralNet_stack_TanH(void* ptr) {
        (cast(NeuralNet)ptr).stack(new TanH());
    }

    void vf_NeuralNet_stack_DropOut(void* ptr, float rate) {
        (cast(NeuralNet)ptr).stack(new DropOut(rate));
    }

    void vf_NeuralNet_stack_SparseKernelExpander(void* ptr, ulong dim_out, char* cross_feats_str, uint max_group_id) {
        string cfs = cast(string)fromStringz(cross_feats_str);
        (cast(NeuralNet)ptr).stack(new SparseKernelExpander(dim_out, cfs, max_group_id));
    }

    // --- Regularization ---

    void vf_NeuralNet_add_regularizer_L1(void* ptr, float lambda_val) {
        auto nn = cast(NeuralNet)ptr;
        if (nn.layers.length == 0) return;
        auto last_layer = nn.layers[$-1];
        if (auto linear = cast(Linear)last_layer) {
            linear.prior(new L1Prior(lambda_val));
        }
    }

    void vf_NeuralNet_add_regularizer_L2(void* ptr, float lambda_val) {
        auto nn = cast(NeuralNet)ptr;
        if (nn.layers.length == 0) return;
        auto last_layer = nn.layers[$-1];
        if (auto linear = cast(Linear)last_layer) {
            linear.prior(new L2Prior(lambda_val));
        }
    }

    void vf_NeuralNet_add_regularizer_Positive(void* ptr, float eps) {
        auto nn = cast(NeuralNet)ptr;
        if (nn.layers.length == 0) return;
        auto last_layer = nn.layers[$-1];
        if (auto linear = cast(Linear)last_layer) {
            linear.prior(new PositivePrior(eps));
        }
    }

    // --- Optimizers ---

    void* vf_Optimizer_create_ADAM(ulong epochs, float lr, ulong batch_size, float eps, float beta1, float beta2) {
        auto opt = new ADAM(epochs, lr, batch_size, eps, beta1, beta2);
        GC.addRoot(cast(void*)opt);
        return cast(void*)opt;
    }

    void* vf_Optimizer_create_AdaGrad(ulong epochs, float lr, ulong batch_size, float eps) {
        auto opt = new AdaGrad(epochs, lr, batch_size, eps);
        GC.addRoot(cast(void*)opt);
        return cast(void*)opt;
    }

    void vf_Optimizer_destroy(void* ptr) {
        if (ptr) {
            GC.removeRoot(ptr);
        }
    }

    // --- Operations ---

    void vf_NeuralNet_initialize(void* ptr, float scale) {
        (cast(NeuralNet)ptr).initialize(scale);
    }

    ulong vf_NeuralNet_num_params(void* ptr) {
        if (ptr is null) return 0;
        return (cast(NeuralNet)ptr).num_params;
    }

    void vf_NeuralNet_serialize(void* ptr, char* path_cstr) {
        string path = cast(string)fromStringz(path_cstr).idup;
        (cast(NeuralNet)ptr).serialize(path);
    }

    void vf_NeuralNet_deserialize(void* ptr, char* path_cstr) {
        string path = cast(string)fromStringz(path_cstr).idup;
        (cast(NeuralNet)ptr).deserialize(path);
    }

    // --- Prediction ---

    // Dense prediction
    void vf_NeuralNet_predict(void* ptr, float* features, ulong n_features, float* out_buffer, ulong n_out) {

        DenseObservation obs;
        obs.features = features[0 .. n_features];

        auto res = (cast(NeuralNet)ptr).predict(obs);

        if (res.length <= n_out) {
            out_buffer[0 .. res.length] = res[];
        }
    }

    // Sparse prediction
    void vf_NeuralNet_predict_sparse(void* ptr, uint* feats_indices, float* feats_values, ulong n_features, float* out_buffer, ulong n_out) {
        SparseObservation obs;
        obs.features = new SparseF[n_features];

        foreach(i; 0..n_features) {
            obs.features[i] = SparseF(feats_indices[i], feats_values[i]);
        }

        auto res = (cast(NeuralNet)ptr).predict(obs);

        if (res.length <= n_out) {
            out_buffer[0 .. res.length] = res[];
        }
    }

    // --- Training ---

    // Dense training
    void vf_NeuralNet_learn(void* ptr,
                            float* X_flat, ulong n_samples, ulong n_features,
                            float* y,
                            void* optimizer_ptr,
                            char* loss_type_cstr,
                            bool verbose, uint num_cores) {

        auto data = new DenseObservation[](n_samples);
        float[] all_features = X_flat[0 .. n_samples * n_features];

        foreach(i; 0..n_samples) {
            data[i].features = all_features[i * n_features .. (i + 1) * n_features];
            data[i].label = y[i];
        }

        auto optimizer = cast(SGDOptimizer)optimizer_ptr;
        string loss_type = cast(string)fromStringz(loss_type_cstr);
        (cast(NeuralNet)ptr).learn(data, loss_type, optimizer, verbose, num_cores);
    }

    // Sparse training
    void vf_NeuralNet_learn_sparse(void* ptr,
                                   float* vals, uint* indices, ulong* indptr,
                                   ulong n_samples,
                                   float* y,
                                   void* optimizer_ptr,
                                   char* loss_type_cstr,
                                   bool verbose, uint num_cores) {

        auto data = new SparseObservation[](n_samples);

        foreach(i; 0..n_samples) {
            ulong start = indptr[i];
            ulong end = indptr[i+1];
            ulong len = end - start;

            data[i].features = new SparseF[len];
            data[i].label = y[i];

            foreach(j; 0..len) {
                ulong idx = start + j;
                data[i].features[j] = SparseF(indices[idx], vals[idx]);
            }
        }

        auto optimizer = cast(SGDOptimizer)optimizer_ptr;
        string loss_type = cast(string)fromStringz(loss_type_cstr);
        (cast(NeuralNet)ptr).learn(data, loss_type, optimizer, verbose, num_cores);
    }
}
