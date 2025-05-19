#pragma once

#include <algorithm> // For std::copy_n, std::fill_n, std::min
#include <ctime>     // For time(nullptr) to seed RNG
#include <random>    // For std::mt19937 and std::uniform_real_distribution
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <string>    // For std::string in exception messages
#include <vector>

#include <sycl/sycl.hpp>        // Main SYCL header
#include <oneapi/math.hpp>      // For oneapi::math::transpose, etc. (used by oneapi::math::blas)
#include <oneapi/math/blas.hpp> // For oneMKL BLAS routines (gemm, asum, nrm2 events)

// ---------------------------------------------------------------------------
// Forward declarations & Enums (Public API - Unchanged)
// ---------------------------------------------------------------------------

/**
 * @enum ActivationType
 * @brief Specifies the type of activation function to be used in a layer.
 */
enum class ActivationType {
    Disabled,  ///< Linear activation (f(x) = x)
    Sigmoid,   ///< Sigmoid activation function
    Tanh,      ///< Hyperbolic tangent activation function
    ReLU,      ///< Rectified Linear Unit activation function
    LeakyReLU, ///< Leaky Rectified Linear Unit activation function
    ELU        ///< Exponential Linear Unit activation function
};

/**
 * @struct LayerDescription
 * @brief Describes the configuration of a single layer in the neural network.
 */
struct LayerDescription {
    unsigned neurons;          ///< Number of neurons in this layer.
    ActivationType activation; ///< Activation function for this layer.
};

/**
 * @struct Regularization
 * @brief Configuration for regularization techniques.
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T> struct Regularization {
    /**
     * @enum Type
     * @brief Type of regularization to apply.
     */
    enum class Type {
        Disabled,  ///< No regularization.
        L1,        ///< L1 regularization (Lasso).
        L2,        ///< L2 regularization (Ridge/Weight Decay).
        ElasticNet ///< Combination of L1 and L2 regularization.
    } type;
    T lambda1{}; ///< Coefficient for L1 regularization.
    T lambda2{}; ///< Coefficient for L2 regularization.

    /**
     * @brief Constructs a Regularization configuration.
     * @param t Type of regularization.
     * @param l1 L1 coefficient.
     * @param l2 L2 coefficient.
     */
    Regularization(Type t = Type::Disabled, T l1 = T(0), T l2 = T(0)) : type(t), lambda1(l1), lambda2(l2) {}
};

/**
 * @struct StopCriteria
 * @brief Configuration for stopping criteria during training.
 * @tparam T The floating-point type.
 */
template <typename T> struct StopCriteria {
    /**
     * @enum Type
     * @brief Type of stopping criterion.
     */
    enum class Type {
        MaxEpochs,     ///< Stop after a maximum number of epochs.
        MinError,      ///< Stop when the total error falls below a threshold.
        MinErrorChange ///< Stop when the change in error between epochs is below a threshold.
    } type;
    T threshold{}; ///< Threshold value for MinError or MinErrorChange.

    /**
     * @brief Constructs a StopCriteria configuration.
     * @param t Type of stopping criterion.
     * @param th Threshold value.
     */
    StopCriteria(Type t = Type::MaxEpochs, T th = T(1e-4)) : type(t), threshold(th) {}
};

/**
 * @struct MomentumConfig
 * @brief Configuration for momentum in gradient descent.
 * @tparam T The floating-point type.
 */
template <typename T> struct MomentumConfig {
    /**
     * @enum Type
     * @brief Type of momentum.
     */
    enum class Type {
        Disabled, ///< No momentum.
        Classical ///< Classical momentum.
    } type;
    T momentum_rate{}; ///< Momentum rate (usually between 0 and 1).

    /**
     * @brief Constructs a MomentumConfig.
     * @param t Type of momentum.
     * @param r Momentum rate.
     */
    MomentumConfig(Type t = Type::Disabled, T r = T(0)) : type(t), momentum_rate(r) {}
};

/**
 * @struct AdaptiveLearningRate
 * @brief Configuration for adaptive learning rate strategies.
 * @tparam T The floating-point type.
 */
template <typename T> struct AdaptiveLearningRate {
    /**
     * @enum Strategy
     * @brief Adaptive learning rate strategy.
     */
    enum class Strategy {
        Constant,    ///< Constant learning rate.
        LinearDecay, ///< Learning rate linearly decays from initial to final_lr over epochs.
        AdaGrad,     ///< AdaGrad adaptive learning rate.
        RMSProp,     ///< RMSProp adaptive learning rate.
        Adam         ///< Adam adaptive learning rate.
    } strategy;
    T epsilon{};  ///< Small constant for numerical stability (e.g., in AdaGrad, RMSProp, Adam).
    T beta1{};    ///< Exponential decay rate for the first moment estimates (Adam, RMSProp rho).
    T beta2{};    ///< Exponential decay rate for the second moment estimates (Adam).
    T final_lr{}; ///< Final learning rate for LinearDecay.

    /**
     * @brief Constructs an AdaptiveLearningRate configuration.
     * @param s Strategy type.
     * @param eps Epsilon value.
     * @param b1 Beta1 value.
     * @param b2 Beta2 value.
     * @param flr Final learning rate for linear decay.
     */
    AdaptiveLearningRate(Strategy s = Strategy::Constant, T eps = T(1e-8), T b1 = T(0.9), T b2 = T(0.999),
                         T flr = T(1e-4))
        : strategy(s), epsilon(eps), beta1(b1), beta2(b2), final_lr(flr) {}
};

/**
 * @enum BackPropagation
 * @brief Specifies the type of backpropagation algorithm (currently only Standard is supported).
 */
enum class BackPropagation { Standard };

// ---------------------------------------------------------------------------
// Activation functions and their derivatives
// These align with f(net_j) and f'(net_j) from the lecture.
// ---------------------------------------------------------------------------

/** @brief Computes the sign of a number. sgn(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0. */
template <typename T> inline constexpr T sgn(T x) noexcept { return T(0) < x ? T(1) : (x < T(0) ? T(-1) : T(0)); }

/** @brief Sigmoid activation function: f(z) = 1 / (1 + exp(-z)). */
template <typename T> inline T sigmoid_activation(T net_input) { return T(1) / (T(1) + sycl::exp(-net_input)); }
/** @brief Derivative of the Sigmoid function: f'(z) = f(z) * (1 - f(z)). */
template <typename T> inline T sigmoid_derivative(T net_input) {
    T s = sigmoid_activation(net_input);
    return s * (T(1) - s);
}

/** @brief Hyperbolic Tangent (tanh) activation function: f(z) = tanh(z). */
template <typename T> inline T tanh_activation(T net_input) { return sycl::tanh(net_input); }
/** @brief Derivative of the tanh function: f'(z) = 1 - tanh^2(z). */
template <typename T> inline T tanh_derivative(T net_input) {
    T t = sycl::tanh(net_input);
    return T(1) - t * t;
}

/** @brief Rectified Linear Unit (ReLU) activation function: f(z) = max(0, z). */
template <typename T> inline T relu_activation(T net_input) { return sycl::fmax(T(0), net_input); }
/** @brief Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0. */
template <typename T> inline T relu_derivative(T net_input) { return net_input > T(0) ? T(1) : T(0); }

/** @brief Leaky ReLU activation function: f(z) = z if z > 0, else alpha*z. */
template <typename T> inline T leaky_relu_activation(T net_input, T alpha = T(0.01)) {
    return net_input > T(0) ? net_input : alpha * net_input;
}
/** @brief Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha. */
template <typename T> inline T leaky_relu_derivative(T net_input, T alpha = T(0.01)) {
    return net_input > T(0) ? T(1) : alpha;
}

/** @brief Exponential Linear Unit (ELU) activation function: f(z) = z if z > 0, else alpha*(exp(z)-1). */
template <typename T> inline T elu_activation(T net_input, T alpha = T(1)) {
    return net_input > T(0) ? net_input : alpha * (sycl::exp(net_input) - T(1));
}
/** @brief Derivative of the ELU function: f'(z) = 1 if z > 0, else alpha*exp(z). */
template <typename T> inline T elu_derivative(T net_input, T alpha = T(1)) {
    return net_input > T(0) ? T(1) : alpha * sycl::exp(net_input);
}

// ---------------------------------------------------------------------------
// USM Device Data Structures
// ---------------------------------------------------------------------------

/**
 * @brief Represents a 2D tensor (matrix) allocated in SYCL USM shared memory.
 * Data is stored in column-major format.
 * @tparam T The data type of tensor elements.
 */
template <typename T> struct DeviceTensor2D {
    T *ptr = nullptr;    ///< Pointer to the USM shared memory.
    std::size_t rows = 0; ///< Number of rows (leading dimension for column-major).
    std::size_t cols = 0; ///< Number of columns.
};

/**
 * @brief Represents a 1D tensor (vector) allocated in SYCL USM shared memory.
 * @tparam T The data type of tensor elements.
 */
template <typename T> struct DeviceTensor1D {
    T *ptr = nullptr;        ///< ptr to the USM shared memory.
    std::size_t elements = 0; ///< Number of elements in the vector.
};

// ---------------------------------------------------------------------------
//  Neural Network class with event-based synchronisation
// ---------------------------------------------------------------------------
/**
 * @class Network
 * @brief A feed-forward neural network implementation using SYCL and oneMKL.
 *
 * This class defines a multi-layer perceptron (MLP) that can be trained
 * using backpropagation with various optimization and regularization options.
 * All computations are performed on a SYCL device using USM shared memory
 * and explicit event-based synchronization for asynchronous execution.
 *
 * @tparam T The floating-point type for network parameters and computations (e.g., float, double).
 */
template <typename T> class Network {
  public:
    /* ========================================================================
     *                            PUBLIC API (Unchanged)
     * ======================================================================== */

    /** @brief Default constructor. Initializes with a default SYCL queue. */
    Network();

    /**
     * @brief Constructs a Network with specified layers, learning parameters, and initial weights/biases.
     * @param layers Vector of LayerDescription defining the network architecture.
     * @param learning_rate Initial learning rate.
     * @param reg Regularization configuration.
     * @param bp Backpropagation algorithm type (currently only Standard).
     * @param adapt Adaptive learning rate strategy configuration.
     * @param stop Stop criteria configuration for training.
     * @param mom Momentum configuration.
     * @param initial_weights Initial weights for each layer.
     * @param initial_biases Initial biases for each layer.
     */
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
            std::vector<std::vector<T>> &initial_weights, std::vector<std::vector<T>> &initial_biases);

    /**
     * @brief Constructs a Network with specified layers, learning parameters, and random weight/bias initialization.
     * @param layers Vector of LayerDescription defining the network architecture.
     * @param learning_rate Initial learning rate.
     * @param reg Regularization configuration.
     * @param bp Backpropagation algorithm type.
     * @param adapt Adaptive learning rate strategy configuration.
     * @param stop Stop criteria configuration for training.
     * @param mom Momentum configuration.
     * @param seed Seed for random number generator for weight/bias initialization.
     */
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed);

    /**
     * @brief Constructs a Network with specified layers, learning parameters, and time-based random seed for
     * weights/biases.
     */
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom);

    /** @brief Destructor. Frees all allocated USM memory. */
    ~Network();

    /**
     * @brief Trains the neural network.
     * @param input_samples Flattened vector of input samples. (NumSamples x NumInputNeurons)
     * @param target_samples Flattened vector of target samples. (NumSamples x NumOutputNeurons)
     * @param num_samples Number of training samples.
     * @param batch_size Size of mini-batches for training.
     * @param max_epochs Maximum number of epochs to train.
     * @return Vector of total loss per epoch.
     */
    std::vector<T> train(const std::vector<T> &input_samples, const std::vector<T> &target_samples,
                         unsigned num_samples, unsigned batch_size, unsigned max_epochs);

    /**
     * @brief Predicts outputs for given input samples.
     * @param input_samples Flattened vector of input samples.
     * @param num_samples Number of input samples.
     * @return Flattened vector of predicted outputs.
     */
    std::vector<T> predict(const std::vector<T> &input_samples, unsigned num_samples);

    /**
     * @brief Retrieves the history of weights and biases at each epoch.
     * @return Pair of vectors of vectors of vectors: {weights_history, biases_history}.
     *         Outer vector is epoch, middle vector is layer, inner vector is flattened weights/biases.
     */
    std::pair<std::vector<std::vector<std::vector<T>>>, std::vector<std::vector<std::vector<T>>>> weights_biases();

    /**
     * @brief Sets the network's weights and biases to specified values.
     * @param new_weights Weights to set.
     * @param new_biases Biases to set.
     */
    void set_weights_biases(const std::vector<std::vector<T>> &new_weights,
                            const std::vector<std::vector<T>> &new_biases);

  private:
    /* ========================================================================
     *                        PRIVATE MEMBERS & HELPERS
     * ======================================================================== */

    /* ---- Immutable Network Configuration ---- */
    std::vector<LayerDescription> m_layer_configs; ///< Configuration for each layer.
    T m_initial_learning_rate{};                   ///< Base learning rate.
    Regularization<T> m_regularization_config;     ///< Regularization parameters.
    MomentumConfig<T> m_momentum_config;           ///< Momentum parameters.
    BackPropagation m_backprop_type;               ///< Type of backpropagation (currently only Standard).
    AdaptiveLearningRate<T> m_adaptive_lr_config;  ///< Adaptive learning rate parameters.
    StopCriteria<T> m_stop_criteria_config;        ///< Training stop criteria.

    /* ---- SYCL Objects ---- */
    sycl::queue m_sycl_queue; ///< SYCL queue for device operations (out-of-order).

    /* ---- Trainable Parameters (Weights and Biases) ---- */
    ///< Weights for each layer (L-1 matrices, W_l connects layer l to l+1). Dimensions: (neurons_l+1, neurons_l).
    std::vector<DeviceTensor2D<T>> m_Weights;
    ///< Biases for each layer (L-1 vectors, B_l for layer l+1). Dimensions: (neurons_l+1).
    std::vector<DeviceTensor1D<T>> m_Biases;

    /* ---- Optimizer State (e.g., for Momentum, Adam) ---- */
    ///< First moment vectors for weights (Adam m_t, Classical momentum velocity).
    std::vector<DeviceTensor2D<T>> m_WeightFirstMoments;
    ///< First moment vectors for biases.
    std::vector<DeviceTensor1D<T>> m_BiasFirstMoments;
    ///< Second moment vectors for weights (Adam v_t, RMSProp E[g^2], AdaGrad sum(g^2)).
    std::vector<DeviceTensor2D<T>> m_WeightSecondMoments;
    ///< Second moment vectors for biases.
    std::vector<DeviceTensor1D<T>> m_BiasSecondMoments;

    /* ---- Per-Batch Training Temporaries (Device Memory) ---- */
    ///< Output activations o_j for each layer j. m_LayerOutputActivations[0] is batch input. (neurons_j, batch_size)
    std::vector<DeviceTensor2D<T>> m_LayerOutputActivations;
    ///< Net inputs net_j for each layer j (before activation). (neurons_j, batch_size)
    std::vector<DeviceTensor2D<T>> m_LayerNetInputs;
    ///< Delta signals δ_j for each layer j. (neurons_j, batch_size)
    std::vector<DeviceTensor2D<T>> m_LayerDeltaSignals;
    ///< Gradients of data loss w.r.t weights ∂E_data/∂W_l. (neurons_l+1, neurons_l)
    std::vector<DeviceTensor2D<T>> m_WeightGradients;
    ///< Gradients of data loss w.r.t biases ∂E_data/∂B_l. (neurons_l+1)
    std::vector<DeviceTensor1D<T>> m_BiasGradients;

    /* ---- Epoch Bookkeeping ---- */
    ///< Accumulated data loss (e.g., MSE) for the current epoch, on device. One element per epoch.
    T *m_EpochDataLossDevice = nullptr;
    ///< Accumulated regularization penalty for the current epoch, on device. One element per epoch.
    T *m_EpochRegPenaltyDevice = nullptr;
    ///< Counter for Adam optimizer steps (1-indexed).
    unsigned m_adamStepCounter{0};
    ///< History of total loss (data loss + regularization penalty) per epoch, stored on host.
    std::vector<T> m_EpochTotalLossHostHistory;

    ///< History of weights at the end of each epoch.
    std::vector<std::vector<std::vector<T>>> m_WeightsHistory;
    ///< History of biases at the end of each epoch.
    std::vector<std::vector<std::vector<T>>> m_BiasesHistory;

    /** @brief Takes a snapshot of current weights and biases and stores them in history. */
    void snapshot_history_();

    /* ---- Helper Routines for Initialization and Memory Management ---- */
    /**
     * @brief Allocates USM memory for network parameters (weights, biases, optimizer states)
     *        and initializes them from provided vectors or randomly. Also takes initial snapshot.
     * @param initial_weights Optional: user-provided initial weights.
     * @param initial_biases Optional: user-provided initial biases.
     * @param seed Optional: seed for random initialization if initial_weights/biases are empty.
     */
    void initialize_parameters_and_history_(const std::vector<std::vector<T>> *initial_weights = nullptr,
                                            const std::vector<std::vector<T>> *initial_biases = nullptr,
                                            unsigned seed = 0);

    /**
     * @brief Allocates USM memory for temporary data structures needed during batch training.
     * @param batch_size The size of the mini-batches.
     * @param num_epochs The maximum number of epochs (for allocating error/norm history arrays).
     */
    void allocate_batch_temporaries_(std::size_t batch_size, std::size_t num_epochs);

    /** @brief Frees USM memory used for batch training temporaries. */
    void deallocate_batch_temporaries_();

    /* ---- Core Backpropagation Algorithm Steps (Private Methods) ---- */

    /**
     * @brief Performs forward propagation for a single layer.
     *        Calculates net_input_next_layer = W_current_layer * output_activation_current_layer + B_current_layer
     *        Then, output_activation_next_layer = activation_function(net_input_next_layer).
     * @param layer_idx Index of the current layer (0 for input layer). Output is for layer_idx+1.
     * @param current_batch_size Number of samples in the current batch.
     * @param dependency_event SYCL event this operation depends on.
     * @return SYCL event for completion of this layer's forward pass.
     */
    sycl::event forward_propagate_layer_(std::size_t layer_idx, std::size_t current_batch_size,
                                         sycl::event dependency_event);

    /**
     * @brief Computes delta signals (δ) for the output layer and accumulates the data loss term.
     *        δ_output = (target_output - actual_output) * f'(net_input_output_layer).
     *        Data Loss (e.g., MSE) = 0.5 * sum((target_output - actual_output)^2).
     * @param target_batch_ptr ptr to the USM device memory for the target outputs of the current batch.
     * @param current_batch_size Number of samples in the current batch.
     * @param current_epoch Index of the current epoch (for storing accumulated error).
     * @param dependency_event SYCL event this operation depends on.
     * @return SYCL event for completion.
     */
    sycl::event compute_output_layer_delta_and_error_(const T *target_batch_ptr, std::size_t current_batch_size,
                                                      unsigned current_epoch, sycl::event dependency_event);

    /**
     * @brief Backpropagates delta signals (δ) from a layer to the preceding hidden layer.
     *        δ_hidden = (W_next_layer^T * δ_next_layer) .* f'(net_input_hidden_layer).
     * @param hidden_layer_idx Index of the hidden layer for which to compute deltas (e.g., if computing for layer j,
     *                         this is j. Deltas from layer j+1 are used).
     *                         Effectively, `m_LayerDeltaSignals[hidden_layer_idx]` is computed using
     * `m_LayerDeltaSignals[hidden_layer_idx+1]`.
     * @param current_batch_size Number of samples in the current batch.
     * @param dependency_event SYCL event this operation depends on.
     * @return SYCL event for completion.
     */
    sycl::event backpropagate_deltas_hidden_layer_(int hidden_layer_idx, std::size_t current_batch_size,
                                                   sycl::event dependency_event);

    /**
     * @brief Computes gradients for weights and biases for a given layer.
     *        ∂E_data/∂W_l = (1/batch_size) * δ_layer(l+1) * (output_activation_layer(l))^T
     *        ∂E_data/∂B_l = (1/batch_size) * sum_batch(δ_layer(l+1))
     * @param layer_idx Index of the layer whose weights/biases connect it to layer_idx+1.
     *                  Gradients are for m_Weights[layer_idx] and m_Biases[layer_idx].
     * @param current_batch_size Number of samples in the current batch.
     * @param dependency_event SYCL event this operation depends on.
     * @return SYCL event for completion.
     */
    sycl::event compute_weight_and_bias_gradients_(std::size_t layer_idx, std::size_t current_batch_size,
                                                   sycl::event dependency_event);

    /**
     * @brief Updates weights and biases for a layer using computed gradients and optimizer rules.
     *        Implements the specific update rule: W_new = W_old + LR_eff * (Gradient_DataLoss -
     * Gradient_Regularization) and similar for biases (typically without regularization).
     * @param layer_idx Index of the layer whose weights/biases are being updated.
     * @param current_batch_size Number of samples in batch (unused by this version but kept for consistency).
     * @param adam_step Current global step count for Adam optimizer (t > 0).
     * @param current_epoch Current epoch number (for learning rate decay).
     * @param max_epochs Total maximum epochs (for learning rate decay).
     * @param gradients_ready_event SYCL event indicating that gradients for this layer are computed.
     * @return SYCL event for completion of parameter updates.
     */
    sycl::event update_weights_and_biases_(std::size_t layer_idx, std::size_t current_batch_size, unsigned adam_step,
                                           unsigned current_epoch, unsigned max_epochs,
                                           sycl::event gradients_ready_event);

    /**
     * @brief Computes and accumulates the L1/L2 regularization penalty for a layer's weights.
     *        Penalty = lambda1 * sum(|W|) + 0.5 * lambda2 * sum(W^2).
     * @param layer_idx Index of the layer whose weights are regularized.
     * @param current_epoch Index of the current epoch (for storing accumulated penalty).
     * @param dependency_event SYCL event this operation depends on.
     * @return SYCL event for completion.
     */
    sycl::event accumulate_regularization_penalty_(std::size_t layer_idx, unsigned current_epoch,
                                                   sycl::event dependency_event);
};

/* ===========================================================================
                                  IMPLEMENTATION
============================================================================ */

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------
template <typename T>
Network<T>::Network()
    : m_sycl_queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{}) {} // Default constructor

template <typename T>
void Network<T>::initialize_parameters_and_history_(const std::vector<std::vector<T>> *initial_weights,
                                                    const std::vector<std::vector<T>> *initial_biases, unsigned seed) {
    // Number of weight/bias sets is NumLayers - 1
    const std::size_t num_parameter_sets = m_layer_configs.size() - 1;
    if (num_parameter_sets == 0) { // Should have at least input and output layer
        throw std::invalid_argument("Network must have at least two layers (input and output).");
    }

    m_Weights.resize(num_parameter_sets);
    m_Biases.resize(num_parameter_sets);
    m_WeightFirstMoments.resize(num_parameter_sets);
    m_BiasFirstMoments.resize(num_parameter_sets);
    m_WeightSecondMoments.resize(num_parameter_sets);
    m_BiasSecondMoments.resize(num_parameter_sets);

    bool use_random_init = (initial_weights == nullptr || initial_biases == nullptr);
    if (!use_random_init) {
        if (initial_weights->size() != num_parameter_sets)
            throw std::invalid_argument("Initial weights vector has incorrect number of layers.");
        if (initial_biases->size() != num_parameter_sets)
            throw std::invalid_argument("Initial biases vector has incorrect number of layers.");
    }

    std::mt19937 random_engine;
    if (use_random_init) {
        random_engine.seed(seed == 0 ? static_cast<unsigned>(time(nullptr)) : seed);
    }
    std::uniform_real_distribution<T> dist(-T(0.1), T(0.1)); // For random init

    for (std::size_t l = 0; l < num_parameter_sets; ++l) {
        std::size_t neurons_current_layer = m_layer_configs[l].neurons;
        std::size_t neurons_next_layer = m_layer_configs[l + 1].neurons;

        if (neurons_current_layer == 0 || neurons_next_layer == 0) {
            throw std::invalid_argument("Layer neuron count must be greater than 0.");
        }

        // Weights W_l connect layer l to layer l+1
        // Dimensions: (neurons_next_layer, neurons_current_layer)
        m_Weights[l].rows = neurons_next_layer;
        m_Weights[l].cols = neurons_current_layer;
        std::size_t weight_size = neurons_next_layer * neurons_current_layer;

        // Biases B_l are for layer l+1
        // Dimensions: (neurons_next_layer)
        m_Biases[l].elements = neurons_next_layer;

        auto allocate_usm = [&](std::size_t n) {
            T *ptr = sycl::malloc_shared<T>(n, m_sycl_queue);
            if (!ptr)
                throw std::runtime_error("Failed to allocate USM shared memory.");
            return ptr;
        };

        m_Weights[l].ptr = allocate_usm(weight_size);
        m_WeightFirstMoments[l].ptr = allocate_usm(weight_size);
        m_WeightSecondMoments[l].ptr = allocate_usm(weight_size);
        m_WeightFirstMoments[l].rows = m_WeightSecondMoments[l].rows = neurons_next_layer;
        m_WeightFirstMoments[l].cols = m_WeightSecondMoments[l].cols = neurons_current_layer;

        m_Biases[l].ptr = allocate_usm(neurons_next_layer);
        m_BiasFirstMoments[l].ptr = allocate_usm(neurons_next_layer);
        m_BiasSecondMoments[l].ptr = allocate_usm(neurons_next_layer);
        m_BiasFirstMoments[l].elements = m_BiasSecondMoments[l].elements = neurons_next_layer;

        if (use_random_init) {
            for (std::size_t i = 0; i < weight_size; ++i)
                m_Weights[l].ptr[i] = dist(random_engine);
            for (std::size_t i = 0; i < neurons_next_layer; ++i)
                m_Biases[l].ptr[i] = dist(random_engine);
        } else {
            if ((*initial_weights)[l].size() != weight_size)
                throw std::invalid_argument("Initial weights size mismatch for layer " + std::to_string(l) + ".");
            if ((*initial_biases)[l].size() != neurons_next_layer)
                throw std::invalid_argument("Initial biases size mismatch for layer " + std::to_string(l) + ".");

            std::copy_n((*initial_weights)[l].data(), weight_size, m_Weights[l].ptr);
            std::copy_n((*initial_biases)[l].data(), neurons_next_layer, m_Biases[l].ptr);
        }

        // Initialize moments to zero
        std::fill_n(m_WeightFirstMoments[l].ptr, weight_size, T(0));
        std::fill_n(m_WeightSecondMoments[l].ptr, weight_size, T(0));
        std::fill_n(m_BiasFirstMoments[l].ptr, neurons_next_layer, T(0));
        std::fill_n(m_BiasSecondMoments[l].ptr, neurons_next_layer, T(0));
    }
    m_sycl_queue.wait_and_throw(); // Ensure initialization is complete on device if host data was copied.
    snapshot_history_();           // Store initial state
}

template <typename T> void Network<T>::snapshot_history_() {
    m_sycl_queue.wait_and_throw(); // Ensure all device operations are finished before reading
    std::vector<std::vector<T>> weights_snapshot(m_Weights.size());
    std::vector<std::vector<T>> biases_snapshot(m_Biases.size());

    for (std::size_t l = 0; l < m_Weights.size(); ++l) {
        weights_snapshot[l].assign(m_Weights[l].ptr,
                                   m_Weights[l].ptr + m_Weights[l].rows * m_Weights[l].cols);
        biases_snapshot[l].assign(m_Biases[l].ptr, m_Biases[l].ptr + m_Biases[l].elements);
    }
    m_WeightsHistory.push_back(std::move(weights_snapshot));
    m_BiasesHistory.push_back(std::move(biases_snapshot));
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt_lr, StopCriteria<T> stop_crit, MomentumConfig<T> mom,
                    std::vector<std::vector<T>> &initial_weights, std::vector<std::vector<T>> &initial_biases)
    : m_layer_configs(std::move(layers)), m_initial_learning_rate(learning_rate), m_regularization_config(reg),
      m_momentum_config(mom), m_backprop_type(bp), m_adaptive_lr_config(adapt_lr), m_stop_criteria_config(stop_crit),
      m_sycl_queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{}) {
    if (m_layer_configs.empty())
        throw std::invalid_argument("Layer configurations cannot be empty.");
    initialize_parameters_and_history_(&initial_weights, &initial_biases);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt_lr, StopCriteria<T> stop_crit, MomentumConfig<T> mom, unsigned seed)
    : m_layer_configs(std::move(layers)), m_initial_learning_rate(learning_rate), m_regularization_config(reg),
      m_momentum_config(mom), m_backprop_type(bp), m_adaptive_lr_config(adapt_lr), m_stop_criteria_config(stop_crit),
      m_sycl_queue(sycl::default_selector_v, sycl::property::queue::enable_profiling{}) {
    if (m_layer_configs.empty())
        throw std::invalid_argument("Layer configurations cannot be empty.");
    initialize_parameters_and_history_(nullptr, nullptr, seed);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt_lr, StopCriteria<T> stop_crit, MomentumConfig<T> mom)
    : Network(std::move(layers), learning_rate, reg, bp, adapt_lr, stop_crit, mom,
              static_cast<unsigned>(time(nullptr))) {}

template <typename T> Network<T>::~Network() {
    m_sycl_queue.wait_and_throw(); // Wait for all pending operations
    auto free_device_tensor_2d_vector = [&](std::vector<DeviceTensor2D<T>> &vec) {
        for (auto &tensor : vec)
            if (tensor.ptr)
                sycl::free(tensor.ptr, m_sycl_queue);
    };
    auto free_device_tensor_1d_vector = [&](std::vector<DeviceTensor1D<T>> &vec) {
        for (auto &tensor : vec)
            if (tensor.ptr)
                sycl::free(tensor.ptr, m_sycl_queue);
    };

    free_device_tensor_2d_vector(m_Weights);
    free_device_tensor_1d_vector(m_Biases);
    free_device_tensor_2d_vector(m_WeightFirstMoments);
    free_device_tensor_1d_vector(m_BiasFirstMoments);
    free_device_tensor_2d_vector(m_WeightSecondMoments);
    free_device_tensor_1d_vector(m_BiasSecondMoments);

    deallocate_batch_temporaries_(); // Frees m_LayerOutputActivations, etc. and epoch-specific device memory

    // m_EpochDataLossDevice and m_EpochRegPenaltyDevice are freed in deallocate_batch_temporaries_
}

// ---------------------------------------------------------------------------
// Batch Temporaries Allocation / Deallocation
// ---------------------------------------------------------------------------
template <typename T> void Network<T>::allocate_batch_temporaries_(std::size_t batch_size, std::size_t num_epochs) {
    if (batch_size == 0)
        throw std::invalid_argument("Batch size must be greater than 0.");
    if (num_epochs == 0)
        throw std::invalid_argument("Number of epochs must be greater than 0.");

    const std::size_t num_layers = m_layer_configs.size();
    auto allocate_usm = [&](std::size_t n) {
        T *ptr = sycl::malloc_shared<T>(n, m_sycl_queue);
        if (!ptr)
            throw std::runtime_error("Failed to allocate USM shared memory for batch temporaries.");
        return ptr;
    };

    m_LayerOutputActivations.resize(num_layers);
    m_LayerNetInputs.resize(num_layers - 1);    // No net input for the input layer
    m_LayerDeltaSignals.resize(num_layers - 1); // No delta signal for the input layer

    for (std::size_t l = 0; l < num_layers; ++l) {
        std::size_t neurons_in_layer_l = m_layer_configs[l].neurons;
        std::size_t num_elements_for_activation_outputs = batch_size * neurons_in_layer_l;

        m_LayerOutputActivations[l].ptr = allocate_usm(num_elements_for_activation_outputs);
        m_LayerOutputActivations[l].rows = neurons_in_layer_l;
        m_LayerOutputActivations[l].cols = batch_size;

        if (l > 0) { // Net inputs and Deltas are for hidden/output layers (l-1 index maps to layer l)
            m_LayerNetInputs[l - 1].ptr = allocate_usm(num_elements_for_activation_outputs);
            m_LayerNetInputs[l - 1].rows = neurons_in_layer_l;
            m_LayerNetInputs[l - 1].cols = batch_size;

            m_LayerDeltaSignals[l - 1].ptr = allocate_usm(num_elements_for_activation_outputs);
            m_LayerDeltaSignals[l - 1].rows = neurons_in_layer_l;
            m_LayerDeltaSignals[l - 1].cols = batch_size;
        }
    }

    m_WeightGradients.resize(num_layers - 1);
    m_BiasGradients.resize(num_layers - 1);
    for (std::size_t l = 0; l < num_layers - 1; ++l) { // Gradients for weights connecting layer l to l+1
        std::size_t neurons_in_layer_l = m_layer_configs[l].neurons;
        std::size_t neurons_in_layer_l_plus_1 = m_layer_configs[l + 1].neurons;

        std::size_t weight_gradient_size = neurons_in_layer_l_plus_1 * neurons_in_layer_l;
        m_WeightGradients[l].ptr = allocate_usm(weight_gradient_size);
        m_WeightGradients[l].rows = neurons_in_layer_l_plus_1;
        m_WeightGradients[l].cols = neurons_in_layer_l;

        m_BiasGradients[l].ptr = allocate_usm(neurons_in_layer_l_plus_1);
        m_BiasGradients[l].elements = neurons_in_layer_l_plus_1;
    }

    m_EpochDataLossDevice = allocate_usm(num_epochs);
    m_EpochRegPenaltyDevice = allocate_usm(num_epochs);
    std::fill_n(m_EpochDataLossDevice, num_epochs, T(0));
    std::fill_n(m_EpochRegPenaltyDevice, num_epochs, T(0));
    m_sycl_queue.wait_and_throw(); // Ensure fills are done.
}

template <typename T> void Network<T>::deallocate_batch_temporaries_() {
    m_sycl_queue.wait_and_throw(); // Ensure all operations using these temporaries are complete
    auto free_device_tensor_2d_vector = [&](std::vector<DeviceTensor2D<T>> &vec) {
        for (auto &tensor : vec)
            if (tensor.ptr)
                sycl::free(tensor.ptr, m_sycl_queue);
        vec.clear();
    };
    auto free_device_tensor_1d_vector = [&](std::vector<DeviceTensor1D<T>> &vec) {
        for (auto &tensor : vec)
            if (tensor.ptr)
                sycl::free(tensor.ptr, m_sycl_queue);
        vec.clear();
    };

    free_device_tensor_2d_vector(m_LayerOutputActivations);
    free_device_tensor_2d_vector(m_LayerNetInputs);
    free_device_tensor_2d_vector(m_LayerDeltaSignals);
    free_device_tensor_2d_vector(m_WeightGradients);
    free_device_tensor_1d_vector(m_BiasGradients);

    if (m_EpochDataLossDevice) {
        sycl::free(m_EpochDataLossDevice, m_sycl_queue);
        m_EpochDataLossDevice = nullptr;
    }
    if (m_EpochRegPenaltyDevice) {
        sycl::free(m_EpochRegPenaltyDevice, m_sycl_queue);
        m_EpochRegPenaltyDevice = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Forward Pass
// ---------------------------------------------------------------------------
template <typename T>
sycl::event Network<T>::forward_propagate_layer_(std::size_t layer_idx, std::size_t current_batch_size,
                                                 sycl::event dependency_event) {
    // W_l (m_Weights[layer_idx]) connects layer l to layer l+1.
    // Input activations are m_LayerOutputActivations[layer_idx].
    // Net inputs and output activations are for layer l+1.
    // m_LayerNetInputs[layer_idx] and m_LayerOutputActivations[layer_idx+1].

    // Dimensions for GEMM: Z = W * In
    // W is (M x K), In is (K x N), Z is (M x N)
    // M = neurons in layer l+1 (m_Weights[layer_idx].rows)
    // K = neurons in layer l   (m_Weights[layer_idx].cols)
    // N = current_batch_size
    const std::size_t M_rows_output = m_Weights[layer_idx].rows;
    const std::size_t K_common_dim = m_Weights[layer_idx].cols;
    const std::size_t N_cols_batch = current_batch_size;

    T *weights_ptr = m_Weights[layer_idx].ptr;
    T *biases_ptr = m_Biases[layer_idx].ptr; // Biases for layer l+1
    T *input_activations_ptr = m_LayerOutputActivations[layer_idx].ptr;
    T *net_inputs_ptr = m_LayerNetInputs[layer_idx].ptr;                     // Net input for layer l+1
    T *output_activations_ptr = m_LayerOutputActivations[layer_idx + 1].ptr; // Output activation for layer l+1

    // Leading dimensions for GEMM (number of rows for column-major matrices)
    const std::size_t lda = M_rows_output; // rows of W
    const std::size_t ldb = K_common_dim;  // rows of InputActivations
    const std::size_t ldc = M_rows_output; // rows of NetInputs

    // Step 1: Compute net input: net_input = W * input_activations (GEMM)
    // Z_MxN = W_MxK * In_KxN
    sycl::event gemm_event = oneapi::math::blas::column_major::gemm(
        m_sycl_queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, M_rows_output, N_cols_batch,
        K_common_dim, T(1), weights_ptr, lda, input_activations_ptr, ldb, T(0), net_inputs_ptr, ldc,
        {dependency_event});

    // Step 2: Apply activation function: output_activation = f(net_input + bias)
    ActivationType activation_func_type = m_layer_configs[layer_idx + 1].activation;
    sycl::event activation_event = m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on(gemm_event); // Depends on GEMM completion
        // Parallelize over batch samples (N_cols_batch) and neurons in the output layer (M_rows_output)
        h.parallel_for(sycl::range<2>(N_cols_batch, M_rows_output), [=](sycl::id<2> item_id) {
            // item_id[0] is batch sample index (n), item_id[1] is neuron index (m)
            std::size_t n_sample_idx = item_id.get(0);
            std::size_t m_neuron_idx = item_id.get(1);
            // Column-major index: row_idx + num_rows * col_idx
            std::size_t flat_idx = m_neuron_idx + M_rows_output * n_sample_idx;

            T current_net_input = net_inputs_ptr[flat_idx] + biases_ptr[m_neuron_idx];
            net_inputs_ptr[flat_idx] = current_net_input; // Store net_input WITH bias for derivative calculation later

            switch (activation_func_type) {
            case ActivationType::Disabled:
                output_activations_ptr[flat_idx] = current_net_input;
                break;
            case ActivationType::Sigmoid:
                output_activations_ptr[flat_idx] = sigmoid_activation(current_net_input);
                break;
            case ActivationType::Tanh:
                output_activations_ptr[flat_idx] = tanh_activation(current_net_input);
                break;
            case ActivationType::ReLU:
                output_activations_ptr[flat_idx] = relu_activation(current_net_input);
                break;
            case ActivationType::LeakyReLU:
                output_activations_ptr[flat_idx] = leaky_relu_activation(current_net_input);
                break;
            case ActivationType::ELU:
                output_activations_ptr[flat_idx] = elu_activation(current_net_input);
                break;
            }
        });
    });
    return activation_event;
}

// ---------------------------------------------------------------------------
// Backward Pass - Delta Calculation
// ---------------------------------------------------------------------------
template <typename T>
sycl::event Network<T>::compute_output_layer_delta_and_error_(const T *target_batch_ptr, std::size_t current_batch_size,
                                                              unsigned current_epoch, sycl::event dependency_event) {

    const std::size_t output_layer_idx_in_configs = m_layer_configs.size() - 1;
    // Deltas, NetInputs are stored with index (layer_config_idx - 1)
    const std::size_t storage_idx_for_output_layer = output_layer_idx_in_configs - 1;

    const std::size_t num_output_neurons = m_layer_configs[output_layer_idx_in_configs].neurons;
    T *delta_signal_output_ptr = m_LayerDeltaSignals[storage_idx_for_output_layer].ptr;
    T *output_activations_ptr = m_LayerOutputActivations[output_layer_idx_in_configs].ptr;
    T *net_inputs_output_ptr =
        m_LayerNetInputs[storage_idx_for_output_layer].ptr; // Net inputs already include bias

    ActivationType activation_func_type = m_layer_configs[output_layer_idx_in_configs].activation;
    T *epoch_data_loss_ptr = m_EpochDataLossDevice; // USM pointer

    return m_sycl_queue.submit([=](sycl::handler &h) {
        h.depends_on(dependency_event);
        h.parallel_for(sycl::range<2>(current_batch_size, num_output_neurons), [=](sycl::id<2> item_id) {
            // item_id[0] is batch sample index (n), item_id[1] is neuron index (k for output layer)
            std::size_t n_sample_idx = item_id.get(0);
            std::size_t k_neuron_idx = item_id.get(1);
            // Column-major index: row_idx + num_rows * col_idx
            std::size_t flat_idx = k_neuron_idx + num_output_neurons * n_sample_idx;

            T net_input_val = net_inputs_output_ptr[flat_idx]; // This net_input includes bias
            T activation_derivative = T(1);                    // Default for Disabled activation

            switch (activation_func_type) {
            case ActivationType::Disabled:
                break; // activation_derivative remains 1
            case ActivationType::Sigmoid:
                activation_derivative = sigmoid_derivative(net_input_val);
                break;
            case ActivationType::Tanh:
                activation_derivative = tanh_derivative(net_input_val);
                break;
            case ActivationType::ReLU:
                activation_derivative = relu_derivative(net_input_val);
                break;
            case ActivationType::LeakyReLU:
                activation_derivative = leaky_relu_derivative(net_input_val);
                break;
            case ActivationType::ELU:
                activation_derivative = elu_derivative(net_input_val);
                break;
            }

            // Error term (target - output)
            T error_term = target_batch_ptr[flat_idx] - output_activations_ptr[flat_idx];

            // Delta signal for output layer: δ_k = (d_k - o_k) * f'_k(net_k)
            delta_signal_output_ptr[flat_idx] = error_term * activation_derivative;

            // Accumulate data loss (e.g., 0.5 * MSE part for this neuron and sample)
            // This needs to be atomic as multiple work-items contribute to the same epoch_data_loss_ptr[current_epoch]
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_loss_accumulator(epoch_data_loss_ptr[current_epoch]);
            atomic_loss_accumulator.fetch_add(T(0.5) * error_term * error_term);
        });
    });
}

template <typename T>
sycl::event Network<T>::backpropagate_deltas_hidden_layer_(
    int hidden_layer_idx_in_storage, // This is the index for m_LayerDeltaSignals, m_LayerNetInputs
                                     // So, m_LayerDeltaSignals[hidden_layer_idx_in_storage] is for layer
                                     // (hidden_layer_idx_in_storage + 1)
    std::size_t current_batch_size, sycl::event dependency_event) {

    // Layer indices:
    // Current hidden layer (for which we calculate delta) is layer_j = hidden_layer_idx_in_storage + 1
    // Next layer (from which deltas are propagated) is layer_k = hidden_layer_idx_in_storage + 2
    // Weights W_kj connect layer_j to layer_k. These are m_Weights[hidden_layer_idx_in_storage + 1]

    // Dimensions for GEMM: D_hidden = W_next^T * D_next
    // W_next^T is (neurons_j x neurons_k)
    // D_next is (neurons_k x batch_size)
    // D_hidden is (neurons_j x batch_size)
    // M_rows_output_gemm = neurons_j
    // K_common_dim_gemm  = neurons_k
    // N_cols_batch_gemm  = current_batch_size
    const std::size_t M_rows_gemm =
        m_Weights[hidden_layer_idx_in_storage + 1].cols; // Neurons in current hidden layer (layer_j)
    const std::size_t K_common_gemm =
        m_Weights[hidden_layer_idx_in_storage + 1].rows; // Neurons in next layer (layer_k)
    const std::size_t N_cols_gemm = current_batch_size;

    T *weights_next_layer_ptr = m_Weights[hidden_layer_idx_in_storage + 1].ptr;
    T *delta_signal_next_layer_ptr = m_LayerDeltaSignals[hidden_layer_idx_in_storage + 1].ptr;
    T *delta_signal_current_hidden_ptr = m_LayerDeltaSignals[hidden_layer_idx_in_storage].ptr;

    // Leading dimensions for GEMM
    const std::size_t lda_gemm = K_common_gemm; // rows of W_next (original, before transpose)
    const std::size_t ldb_gemm = K_common_gemm; // rows of Delta_next
    const std::size_t ldc_gemm = M_rows_gemm;   // rows of Delta_hidden (output of GEMM)

    // Step 1: Summation part: sum_k (δ_k * w_kj) using GEMM: D_hidden = W_next^T * D_next
    sycl::event gemm_event = oneapi::math::blas::column_major::gemm(
        m_sycl_queue, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, M_rows_gemm, N_cols_gemm,
        K_common_gemm, T(1), weights_next_layer_ptr, lda_gemm, delta_signal_next_layer_ptr, ldb_gemm, T(0),
        delta_signal_current_hidden_ptr, ldc_gemm, {dependency_event});

    // Step 2: Element-wise multiplication with f'(net_input_current_hidden)
    T *net_inputs_current_hidden_ptr =
        m_LayerNetInputs[hidden_layer_idx_in_storage].ptr; // Net inputs already include bias
    ActivationType activation_func_type = m_layer_configs[hidden_layer_idx_in_storage + 1].activation;
    const std::size_t num_neurons_current_hidden = M_rows_gemm; // Same as ldc_gemm

    sycl::event activation_deriv_event = m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on(gemm_event);
        h.parallel_for(sycl::range<2>(current_batch_size, num_neurons_current_hidden), [=](sycl::id<2> item_id) {
            // item_id[0] is batch sample index (n), item_id[1] is neuron index (j for current hidden layer)
            std::size_t n_sample_idx = item_id.get(0);
            std::size_t j_neuron_idx = item_id.get(1);
            std::size_t flat_idx = j_neuron_idx + num_neurons_current_hidden * n_sample_idx;

            T net_input_val = net_inputs_current_hidden_ptr[flat_idx]; // Includes bias
            T activation_derivative = T(1);

            switch (activation_func_type) {
            case ActivationType::Disabled:
                break;
            case ActivationType::Sigmoid:
                activation_derivative = sigmoid_derivative(net_input_val);
                break;
            case ActivationType::Tanh:
                activation_derivative = tanh_derivative(net_input_val);
                break;
            case ActivationType::ReLU:
                activation_derivative = relu_derivative(net_input_val);
                break;
            case ActivationType::LeakyReLU:
                activation_derivative = leaky_relu_derivative(net_input_val);
                break;
            case ActivationType::ELU:
                activation_derivative = elu_derivative(net_input_val);
                break;
            }
            delta_signal_current_hidden_ptr[flat_idx] *= activation_derivative;
        });
    });
    return activation_deriv_event;
}

// ---------------------------------------------------------------------------
// Backward Pass - Gradient Accumulation
// ---------------------------------------------------------------------------
template <typename T>
sycl::event Network<T>::compute_weight_and_bias_gradients_(
    std::size_t layer_idx, // Gradients for m_Weights[layer_idx] and m_Biases[layer_idx]
    std::size_t current_batch_size, sycl::event dependency_event) {

    // layer_idx connects layer_l (input to these weights) to layer_l_plus_1 (output of these weights)
    // Deltas used are from layer_l_plus_1: m_LayerDeltaSignals[layer_idx]
    // Activations used are from layer_l: m_LayerOutputActivations[layer_idx]

    // Dimensions for Weight Gradient GEMM: GW = Delta_next * Activations_current^T
    // Delta_next is (neurons_l+1 x batch_size)
    // Activations_current^T is (batch_size x neurons_l)
    // GW is (neurons_l+1 x neurons_l)
    // M_rows_gw = neurons_l+1
    // K_common_gw = batch_size
    // N_cols_gw = neurons_l
    const std::size_t M_rows_gw = m_LayerDeltaSignals[layer_idx].rows; // Neurons in layer l+1
    const std::size_t K_common_gw = current_batch_size;
    const std::size_t N_cols_gw = m_LayerOutputActivations[layer_idx].rows; // Neurons in layer l

    T *delta_next_ptr = m_LayerDeltaSignals[layer_idx].ptr;
    T *activations_current_ptr = m_LayerOutputActivations[layer_idx].ptr;
    T *weight_gradients_ptr = m_WeightGradients[layer_idx].ptr;
    T *bias_gradients_ptr = m_BiasGradients[layer_idx].ptr;

    // Leading dimensions for GEMM
    const std::size_t lda_gw = M_rows_gw; // rows of Delta_next
    const std::size_t ldb_gw = N_cols_gw; // rows of Activations_current (original, before transpose)
    const std::size_t ldc_gw = M_rows_gw; // rows of GW

    // Step 1: Compute Weight Gradients: GW = (1/batch_size) * Delta_next * Activations_current^T
    sycl::event weight_grad_event =
        oneapi::math::blas::column_major::gemm(m_sycl_queue, oneapi::math::transpose::nontrans,
                                               oneapi::math::transpose::trans, M_rows_gw, N_cols_gw, K_common_gw,
                                               T(1) / static_cast<T>(current_batch_size), // Alpha scaling factor
                                               delta_next_ptr, lda_gw, activations_current_ptr, ldb_gw, T(0),
                                               weight_gradients_ptr, ldc_gw, {dependency_event});

    // Step 2: Compute Bias Gradients: GB_j = (1/batch_size) * sum_over_batch_samples(Delta_next_j_sample)
    // This is equivalent to row-summing Delta_next and scaling.
    // Or, sum each row of Delta_next (which is M_rows_gw x K_common_gw)
    const std::size_t num_neurons_for_bias = M_rows_gw; // Bias gradients for layer l+1
    sycl::event bias_grad_event = m_sycl_queue.submit([&](sycl::handler &h) {
        // Bias gradient calculation can also depend on the delta signals being ready,
        // which is covered by dependency_event passed to GEMM.
        // To be safe, explicitly depend on dependency_event if not on weight_grad_event.
        // Here, it logically follows delta computation, so dependency_event is the key.
        h.depends_on(dependency_event);
        h.parallel_for(sycl::range<1>(num_neurons_for_bias), [=](sycl::id<1> j_neuron_id_obj) {
            std::size_t j_neuron_idx = j_neuron_id_obj.get(0);
            T sum_deltas_for_neuron_j = T(0);
            for (std::size_t n_sample_idx = 0; n_sample_idx < current_batch_size; ++n_sample_idx) {
                // Delta_next is (num_neurons_for_bias x current_batch_size)
                // flat_idx = row_idx + num_rows * col_idx
                sum_deltas_for_neuron_j += delta_next_ptr[j_neuron_idx + num_neurons_for_bias * n_sample_idx];
            }
            bias_gradients_ptr[j_neuron_idx] = sum_deltas_for_neuron_j / static_cast<T>(current_batch_size);
        });
    });

    // Return a combined event. The bias gradient kernel can run concurrently with GEMM if resources allow,
    // but both depend on `dependency_event`. For simplicity in chaining, let's make one depend on other or join.
    // A barrier task is cleaner for joining.
    return m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on({weight_grad_event, bias_grad_event});
        h.single_task([] {}); // Empty task to create a join event
    });
}

// ---------------------------------------------------------------------------
// Parameter Update
// ---------------------------------------------------------------------------
template <typename T>
sycl::event Network<T>::accumulate_regularization_penalty_(std::size_t layer_idx, unsigned current_epoch,
                                                           sycl::event dependency_event) {

    const bool needs_l1 = (m_regularization_config.type == Regularization<T>::Type::L1 ||
                           m_regularization_config.type == Regularization<T>::Type::ElasticNet);
    const bool needs_l2 = (m_regularization_config.type == Regularization<T>::Type::L2 ||
                           m_regularization_config.type == Regularization<T>::Type::ElasticNet);

    if (!needs_l1 && !needs_l2) {
        return dependency_event; // No regularization, pass through dependency
    }

    std::size_t num_weights = m_Weights[layer_idx].rows * m_Weights[layer_idx].cols;
    if (num_weights == 0)
        return dependency_event;

    // Temporary USM scalars for asum/nrm2 results
    T *l1_penalty_device_temp = nullptr;
    T *l2_norm_device_temp = nullptr; // nrm2 returns sqrt(sum(W^2))

    if (needs_l1)
        l1_penalty_device_temp = sycl::malloc_shared<T>(1, m_sycl_queue);
    if (needs_l2)
        l2_norm_device_temp = sycl::malloc_shared<T>(1, m_sycl_queue);

    sycl::event l1_event, l2_event; // Default constructed events

    if (needs_l1) {
        l1_event = oneapi::math::blas::column_major::asum(m_sycl_queue, num_weights, m_Weights[layer_idx].ptr, 1,
                                                          l1_penalty_device_temp, {dependency_event});
    }
    if (needs_l2) {
        l2_event = oneapi::math::blas::column_major::nrm2(m_sycl_queue, num_weights, m_Weights[layer_idx].ptr, 1,
                                                          l2_norm_device_temp, {dependency_event});
    }

    // Values captured by the kernel lambda
    const T lambda1_coeff = m_regularization_config.lambda1;
    const T lambda2_coeff = m_regularization_config.lambda2;
    T *epoch_reg_penalty_ptr = m_EpochRegPenaltyDevice; // USM pointer

    return m_sycl_queue.submit([=](sycl::handler &h) {
        if (needs_l1)
            h.depends_on(l1_event);
        if (needs_l2)
            h.depends_on(l2_event);
        // Also ensure it depends on the original dependency_event if L1/L2 events are not created (e.g. num_weights=0)
        // However, asum/nrm2 already depend on it.

        h.single_task([=] {
            T penalty_contribution = T(0);
            if (needs_l1) {
                penalty_contribution += lambda1_coeff * (*l1_penalty_device_temp);
            }
            if (needs_l2) {
                T norm_val = *l2_norm_device_temp;
                penalty_contribution += T(0.5) * lambda2_coeff * norm_val * norm_val;
            }

            // Atomically add this layer's penalty to the total for the epoch
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_penalty_accumulator(epoch_reg_penalty_ptr[current_epoch]);
            atomic_penalty_accumulator.fetch_add(penalty_contribution);
        });
    });
}

template <typename T>
sycl::event Network<T>::update_weights_and_biases_(
    std::size_t layer_idx, [[maybe_unused]] std::size_t current_batch_size, // current_batch_size unused in this version
    unsigned adam_step, unsigned current_epoch, unsigned max_epochs, sycl::event gradients_ready_event) {

    // Captured parameters from member variables for lambda
    const auto adapt_lr_strategy = m_adaptive_lr_config.strategy;
    const auto reg_type = m_regularization_config.type;
    const T base_learning_rate = m_initial_learning_rate;
    const T adapt_lr_epsilon = m_adaptive_lr_config.epsilon;
    const T adapt_lr_beta1 = m_adaptive_lr_config.beta1; // Adam beta1, RMSProp rho
    const T adapt_lr_beta2 = m_adaptive_lr_config.beta2; // Adam beta2
    const T final_lr_for_decay = m_adaptive_lr_config.final_lr;
    const T momentum_update_rate = m_momentum_config.momentum_rate;
    const bool use_classical_momentum = (m_momentum_config.type == MomentumConfig<T>::Type::Classical);
    const T lambda1_coeff = m_regularization_config.lambda1;
    const T lambda2_coeff = m_regularization_config.lambda2;

    const T linear_decay_gamma =
        (max_epochs > 1) ? (static_cast<T>(current_epoch) / static_cast<T>(max_epochs - 1)) : T(0);

    // ptrs to device data
    T *weights_ptr = m_Weights[layer_idx].ptr;
    T *weight_gradients_ptr = m_WeightGradients[layer_idx].ptr;
    T *weight_first_moment_ptr = m_WeightFirstMoments[layer_idx].ptr;
    T *weight_second_moment_ptr = m_WeightSecondMoments[layer_idx].ptr;
    const std::size_t num_weights = m_Weights[layer_idx].rows * m_Weights[layer_idx].cols;

    T *biases_ptr = m_Biases[layer_idx].ptr;
    T *bias_gradients_ptr = m_BiasGradients[layer_idx].ptr;
    T *bias_first_moment_ptr = m_BiasFirstMoments[layer_idx].ptr;
    T *bias_second_moment_ptr = m_BiasSecondMoments[layer_idx].ptr;
    const std::size_t num_biases = m_Biases[layer_idx].elements;

    // Kernel for updating weights
    sycl::event weights_update_event = m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on(gradients_ready_event);
        h.parallel_for(sycl::range<1>(num_weights), [=](sycl::id<1> item_id) {
            std::size_t i = item_id.get(0);
            T current_weight_value = weights_ptr[i];
            T data_loss_gradient = weight_gradients_ptr[i]; // This is ∂L_data/∂W_i

            T regularization_gradient = T(0);
            if (reg_type == Regularization<T>::Type::L1) {
                regularization_gradient = lambda1_coeff * sgn(current_weight_value);
            } else if (reg_type == Regularization<T>::Type::L2) {
                regularization_gradient = lambda2_coeff * current_weight_value;
            } else if (reg_type == Regularization<T>::Type::ElasticNet) {
                regularization_gradient =
                    lambda1_coeff * sgn(current_weight_value) + lambda2_coeff * current_weight_value;
            }

            // The original code implemented: W_new = W_old + LR_eff * (DataGrad - RegGrad)
            // So, the "effective gradient" for adaptive methods (that will be scaled and ADDED) is (DataGrad - RegGrad)
            T combined_term_for_update = data_loss_gradient - regularization_gradient;

            T learning_step_magnitude = T(0); // This will be (effective_lr * combined_term_for_update) or similar
            T current_effective_lr = base_learning_rate;

            switch (adapt_lr_strategy) {
            case AdaptiveLearningRate<T>::Strategy::Constant:
                if (use_classical_momentum) {
                    weight_first_moment_ptr[i] = momentum_update_rate * weight_first_moment_ptr[i] +
                                                 current_effective_lr * combined_term_for_update;
                    learning_step_magnitude = weight_first_moment_ptr[i];
                } else {
                    learning_step_magnitude = current_effective_lr * combined_term_for_update;
                }
                break;

            case AdaptiveLearningRate<T>::Strategy::LinearDecay:
                current_effective_lr =
                    base_learning_rate * (T(1) - linear_decay_gamma) + linear_decay_gamma * final_lr_for_decay;
                if (use_classical_momentum) {
                    weight_first_moment_ptr[i] = momentum_update_rate * weight_first_moment_ptr[i] +
                                                 current_effective_lr * combined_term_for_update;
                    learning_step_magnitude = weight_first_moment_ptr[i];
                } else {
                    learning_step_magnitude = current_effective_lr * combined_term_for_update;
                }
                break;

            case AdaptiveLearningRate<T>::Strategy::AdaGrad:
                // Note: AdaGrad uses the gradient itself, not (DataGrad - RegGrad) if that was a special combined term.
                // The original code's "positive_total_grad" was (-DataGrad + RegGrad).
                // AdaGrad sums squares of this "positive_total_grad".
                // If combined_term_for_update = (DataGrad - RegGrad), then "positive_total_grad" =
                // -combined_term_for_update. Let's use G_eff = (-DataGrad + RegGrad) as the term whose squares are
                // accumulated, consistent with original.
                {
                    T G_eff = -data_loss_gradient + regularization_gradient; // Term used in original for accumulation
                    weight_second_moment_ptr[i] += G_eff * G_eff;
                    current_effective_lr =
                        base_learning_rate / (sycl::sqrt(weight_second_moment_ptr[i]) + adapt_lr_epsilon);
                    // Original update was W = W - LR_eff * G_eff.
                    // So W = W - LR_eff * (-DataGrad + RegGrad) = W + LR_eff * (DataGrad - RegGrad)
                    learning_step_magnitude = current_effective_lr * combined_term_for_update;
                }
                break;

            case AdaptiveLearningRate<T>::Strategy::RMSProp: {
                T G_eff = -data_loss_gradient + regularization_gradient;
                weight_second_moment_ptr[i] =
                    adapt_lr_beta1 * weight_second_moment_ptr[i] + (T(1) - adapt_lr_beta1) * G_eff * G_eff;
                current_effective_lr =
                    base_learning_rate / (sycl::sqrt(weight_second_moment_ptr[i]) + adapt_lr_epsilon);
                learning_step_magnitude = current_effective_lr * combined_term_for_update;
            } break;

            case AdaptiveLearningRate<T>::Strategy::Adam: {
                T G_eff = -data_loss_gradient + regularization_gradient; // This is `positive_total_grad` from original
                if (adam_step > 0) {                                     // adam_step is 1-indexed
                    weight_first_moment_ptr[i] =
                        adapt_lr_beta1 * weight_first_moment_ptr[i] + (T(1) - adapt_lr_beta1) * G_eff;
                    weight_second_moment_ptr[i] =
                        adapt_lr_beta2 * weight_second_moment_ptr[i] + (T(1) - adapt_lr_beta2) * G_eff * G_eff;

                    T m_hat =
                        weight_first_moment_ptr[i] / (T(1) - sycl::pow(adapt_lr_beta1, static_cast<T>(adam_step)));
                    T v_hat =
                        weight_second_moment_ptr[i] / (T(1) - sycl::pow(adapt_lr_beta2, static_cast<T>(adam_step)));

                    // Original update was W = W - LR * m_hat / (sqrt(v_hat)+eps)
                    // This means the term subtracted is based on G_eff.
                    // So to get W = W + LR_eff * (DataGrad - RegGrad), we need:
                    // LR_eff * (DataGrad - RegGrad) = - [ LR * m_hat / (sqrt(v_hat)+eps) ] where m_hat,v_hat are from
                    // G_eff
                    learning_step_magnitude = -(base_learning_rate * m_hat / (sycl::sqrt(v_hat) + adapt_lr_epsilon));
                } else { // Fallback for adam_step = 0 (should not happen if adam_step is 1-indexed)
                    learning_step_magnitude = base_learning_rate * combined_term_for_update;
                }
            } break;
            }
            // Final update: W_new = W_old + learning_step_magnitude
            // where learning_step_magnitude is effectively LR_eff * (DataGrad - RegGrad)
            weights_ptr[i] = current_weight_value + learning_step_magnitude;
        });
    });

    // Kernel for updating biases (typically no regularization)
    sycl::event biases_update_event = m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on(gradients_ready_event);
        h.parallel_for(sycl::range<1>(num_biases), [=](sycl::id<1> item_id) {
            std::size_t j = item_id.get(0);
            T current_bias_value = biases_ptr[j];
            T data_loss_gradient_bias = bias_gradients_ptr[j]; // This is ∂L_data/∂B_j

            // Biases usually aren't regularized. So combined_term_for_update is just data_loss_gradient_bias.
            T combined_term_for_update_bias = data_loss_gradient_bias;

            T learning_step_magnitude_bias = T(0);
            T current_effective_lr_bias = base_learning_rate;

            switch (adapt_lr_strategy) {
            case AdaptiveLearningRate<T>::Strategy::Constant:
                if (use_classical_momentum) {
                    bias_first_moment_ptr[j] = momentum_update_rate * bias_first_moment_ptr[j] +
                                               current_effective_lr_bias * combined_term_for_update_bias;
                    learning_step_magnitude_bias = bias_first_moment_ptr[j];
                } else {
                    learning_step_magnitude_bias = current_effective_lr_bias * combined_term_for_update_bias;
                }
                break;
            case AdaptiveLearningRate<T>::Strategy::LinearDecay:
                current_effective_lr_bias =
                    base_learning_rate * (T(1) - linear_decay_gamma) + linear_decay_gamma * final_lr_for_decay;
                if (use_classical_momentum) {
                    bias_first_moment_ptr[j] = momentum_update_rate * bias_first_moment_ptr[j] +
                                               current_effective_lr_bias * combined_term_for_update_bias;
                    learning_step_magnitude_bias = bias_first_moment_ptr[j];
                } else {
                    learning_step_magnitude_bias = current_effective_lr_bias * combined_term_for_update_bias;
                }
                break;
            case AdaptiveLearningRate<T>::Strategy::AdaGrad: {
                T G_eff_bias = -data_loss_gradient_bias; // Term used in original for accumulation
                bias_second_moment_ptr[j] += G_eff_bias * G_eff_bias;
                current_effective_lr_bias =
                    base_learning_rate / (sycl::sqrt(bias_second_moment_ptr[j]) + adapt_lr_epsilon);
                learning_step_magnitude_bias = current_effective_lr_bias * combined_term_for_update_bias;
            } break;
            case AdaptiveLearningRate<T>::Strategy::RMSProp: {
                T G_eff_bias = -data_loss_gradient_bias;
                bias_second_moment_ptr[j] =
                    adapt_lr_beta1 * bias_second_moment_ptr[j] + (T(1) - adapt_lr_beta1) * G_eff_bias * G_eff_bias;
                current_effective_lr_bias =
                    base_learning_rate / (sycl::sqrt(bias_second_moment_ptr[j]) + adapt_lr_epsilon);
                learning_step_magnitude_bias = current_effective_lr_bias * combined_term_for_update_bias;
            } break;
            case AdaptiveLearningRate<T>::Strategy::Adam: {
                T G_eff_bias = -data_loss_gradient_bias;
                if (adam_step > 0) {
                    bias_first_moment_ptr[j] =
                        adapt_lr_beta1 * bias_first_moment_ptr[j] + (T(1) - adapt_lr_beta1) * G_eff_bias;
                    bias_second_moment_ptr[j] =
                        adapt_lr_beta2 * bias_second_moment_ptr[j] + (T(1) - adapt_lr_beta2) * G_eff_bias * G_eff_bias;

                    T m_hat_bias =
                        bias_first_moment_ptr[j] / (T(1) - sycl::pow(adapt_lr_beta1, static_cast<T>(adam_step)));
                    T v_hat_bias =
                        bias_second_moment_ptr[j] / (T(1) - sycl::pow(adapt_lr_beta2, static_cast<T>(adam_step)));

                    learning_step_magnitude_bias =
                        -(base_learning_rate * m_hat_bias / (sycl::sqrt(v_hat_bias) + adapt_lr_epsilon));
                } else {
                    learning_step_magnitude_bias = base_learning_rate * combined_term_for_update_bias;
                }
            } break;
            }
            biases_ptr[j] = current_bias_value + learning_step_magnitude_bias;
        });
    });

    // Return a combined event indicating both weight and bias updates are submitted
    return m_sycl_queue.submit([&](sycl::handler &h) {
        h.depends_on({weights_update_event, biases_update_event});
        h.single_task([] {}); // Empty task to create a join event
    });
}

// ---------------------------------------------------------------------------
// Training Loop
// ---------------------------------------------------------------------------
template <typename T>
std::vector<T> Network<T>::train(const std::vector<T> &input_samples, const std::vector<T> &target_samples,
                                 unsigned num_samples, unsigned batch_size, unsigned max_epochs_param) {
    if (m_layer_configs.empty())
        throw std::runtime_error("Network not initialized or has no layers.");
    if (num_samples == 0)
        throw std::invalid_argument("Number of samples cannot be zero.");
    if (batch_size == 0)
        throw std::invalid_argument("Batch size cannot be zero.");
    if (max_epochs_param == 0 && m_stop_criteria_config.type != StopCriteria<T>::Type::MaxEpochs) {
        // If max_epochs is 0 but another stop criterion is primary, it might be intended.
        // However, if MaxEpochs is the only criterion, 0 makes no sense.
        // The original code implies max_epochs is also a hard limit.
        if (m_stop_criteria_config.type == StopCriteria<T>::Type::MaxEpochs)
            throw std::invalid_argument("Max epochs must be greater than 0 if it's the stop criterion.");
    }
    unsigned max_epochs = (max_epochs_param == 0) ? 1 : max_epochs_param; // Ensure at least 1 epoch if 0 passed.

    if (input_samples.size() != num_samples * m_layer_configs.front().neurons)
        throw std::runtime_error("Input samples vector size mismatch with num_samples and input layer neuron count.");
    if (target_samples.size() != num_samples * m_layer_configs.back().neurons)
        throw std::runtime_error("Target samples vector size mismatch with num_samples and output layer neuron count.");

    allocate_batch_temporaries_(batch_size, max_epochs);
    m_EpochTotalLossHostHistory.assign(max_epochs, T(0)); // Initialize with zeros, will be resized if early stop
    m_adamStepCounter = 0;                                // Reset Adam step counter for a new training session

    // Copy full dataset to USM shared memory for efficient batch access
    T *full_input_samples_device = sycl::malloc_shared<T>(input_samples.size(), m_sycl_queue);
    T *full_target_samples_device = sycl::malloc_shared<T>(target_samples.size(), m_sycl_queue);
    if (!full_input_samples_device || !full_target_samples_device)
        throw std::runtime_error("Failed to allocate USM for full dataset.");

    // Asynchronously copy data, then wait.
    sycl::event copy_input_event =
        m_sycl_queue.memcpy(full_input_samples_device, input_samples.data(), input_samples.size() * sizeof(T));
    sycl::event copy_target_event =
        m_sycl_queue.memcpy(full_target_samples_device, target_samples.data(), target_samples.size() * sizeof(T));
    sycl::event::wait_and_throw({copy_input_event, copy_target_event});

    unsigned actual_epochs_run = 0;
    for (unsigned epoch = 0; epoch < max_epochs; ++epoch) {
        actual_epochs_run++;
        sycl::event
            epoch_operations_chain_event; // Start with a default (completed) event or handle first batch carefully

        for (unsigned batch_start_idx = 0; batch_start_idx < num_samples; batch_start_idx += batch_size) {
            unsigned current_batch_size = std::min(batch_size, num_samples - batch_start_idx);

            if (m_adaptive_lr_config.strategy == AdaptiveLearningRate<T>::Strategy::Adam) {
                m_adamStepCounter++; // Increment for Adam (1-indexed for formulas)
            }

            sycl::event current_batch_processing_event = epoch_operations_chain_event; // Chain from previous batch

            // --- Asynchronously copy current batch input to device working memory ---
            // m_LayerOutputActivations[0] serves as input buffer for the first layer
            current_batch_processing_event =
                m_sycl_queue.memcpy(m_LayerOutputActivations[0].ptr,
                                    full_input_samples_device + batch_start_idx * m_layer_configs[0].neurons,
                                    sizeof(T) * current_batch_size * m_layer_configs[0].neurons,
                                    {current_batch_processing_event}); // Dependency

            // --- Forward Pass ---
            for (std::size_t l = 0; l < m_layer_configs.size() - 1; ++l) {
                current_batch_processing_event =
                    forward_propagate_layer_(l, current_batch_size, current_batch_processing_event);
            }

            // --- Output Layer Delta and Error Calculation ---
            const T *target_batch_device_ptr =
                full_target_samples_device + batch_start_idx * m_layer_configs.back().neurons;
            current_batch_processing_event = compute_output_layer_delta_and_error_(
                target_batch_device_ptr, current_batch_size, epoch, current_batch_processing_event);

            // --- Backward Pass - Hidden Layer Deltas ---
            // Iterate from second to last layer (index m_layer_configs.size()-2) down to input layer's preceding one.
            // Storage index for deltas/netinputs is (layer_config_idx - 1).
            // So, if layer_configs has L items (0 to L-1), storage is 0 to L-2.
            // Output layer is L-1 (storage L-2). Hidden layers go down to storage index 0 (layer 1).
            for (int l_storage_idx = static_cast<int>(m_layer_configs.size()) - 3; l_storage_idx >= 0;
                 --l_storage_idx) {
                current_batch_processing_event = backpropagate_deltas_hidden_layer_(l_storage_idx, current_batch_size,
                                                                                    current_batch_processing_event);
            }

            // --- Gradient Computation ---
            // Gradients for weights W_l (connecting layer l to l+1)
            std::vector<sycl::event> gradient_computation_events(m_layer_configs.size() - 1);
            for (std::size_t l = 0; l < m_layer_configs.size() - 1; ++l) {
                gradient_computation_events[l] =
                    compute_weight_and_bias_gradients_(l, current_batch_size, current_batch_processing_event);
            }
            // Create a single event that depends on all gradient computations for this batch
            sycl::event all_gradients_ready_event = m_sycl_queue.submit([&](sycl::handler &h) {
                h.depends_on(gradient_computation_events);
                h.single_task([] {});
            });

            // --- Parameter Updates ---
            std::vector<sycl::event> parameter_update_events(m_layer_configs.size() - 1);
            unsigned step_for_adam_update =
                (m_adaptive_lr_config.strategy == AdaptiveLearningRate<T>::Strategy::Adam) ? m_adamStepCounter : 0;
            for (std::size_t l = 0; l < m_layer_configs.size() - 1; ++l) {
                parameter_update_events[l] = update_weights_and_biases_(
                    l, current_batch_size, step_for_adam_update, epoch, max_epochs,
                    all_gradients_ready_event); // Each update depends on all gradients being ready
                                                // (though technically only its own layer's gradient)
                                                // For simplicity and to ensure correct chaining, using combined event.
            }
            // Update the chain event for the next batch
            epoch_operations_chain_event = m_sycl_queue.submit([&](sycl::handler &h) {
                h.depends_on(parameter_update_events);
                h.single_task([] {});
            });
        } // End of batch loop

        // Wait for all batch operations in the current epoch to complete before regularization penalty calculation
        epoch_operations_chain_event.wait_and_throw();

        // --- Regularization Penalty Calculation (after all weights for the epoch are updated) ---
        sycl::event regularization_penalty_event = epoch_operations_chain_event; // Start after previous epoch ops
        const bool needs_regularization = (m_regularization_config.type != Regularization<T>::Type::Disabled);
        if (needs_regularization) {
            std::vector<sycl::event> layer_penalty_events(m_layer_configs.size() - 1);
            for (std::size_t l = 0; l < m_layer_configs.size() - 1; ++l) {
                layer_penalty_events[l] = accumulate_regularization_penalty_(l, epoch, epoch_operations_chain_event);
            }
            regularization_penalty_event = m_sycl_queue.submit([&](sycl::handler &h) {
                h.depends_on(layer_penalty_events);
                h.single_task([] {});
            });
        } else {
            // If no regularization, m_EpochRegPenaltyDevice[epoch] is already 0 (from alloc_train_)
            // Ensure previous operations are waited upon if we were to write to it from host.
            // Here, it's already 0 and on device, so just pass the event.
        }

        regularization_penalty_event.wait_and_throw(); // Wait for penalty calculations (if any)

        // --- Compute Total Loss for the Epoch (on host, after device data is ready) ---
        // m_EpochDataLossDevice and m_EpochRegPenaltyDevice are USM shared, values are atomically updated.
        // The wait_and_throw above ensures device computations are complete.
        T current_epoch_data_loss = m_EpochDataLossDevice[epoch];
        T current_epoch_reg_penalty = m_EpochRegPenaltyDevice[epoch];
        m_EpochTotalLossHostHistory[epoch] =
            (current_epoch_data_loss / static_cast<T>(num_samples)) + current_epoch_reg_penalty; // Average data loss

        snapshot_history_(); // snapshot_history_ itself calls m_sycl_queue.wait_and_throw()

        // --- Early Stopping Check ---
        bool stop_training_flag = false;
        if (m_stop_criteria_config.type == StopCriteria<T>::Type::MinError) {
            if (m_EpochTotalLossHostHistory[epoch] < m_stop_criteria_config.threshold)
                stop_training_flag = true;
        } else if (m_stop_criteria_config.type == StopCriteria<T>::Type::MinErrorChange) {
            if (epoch > 0 && sycl::fabs(m_EpochTotalLossHostHistory[epoch - 1] - m_EpochTotalLossHostHistory[epoch]) <
                                 m_stop_criteria_config.threshold)
                stop_training_flag = true;
        }
        // MaxEpochs is handled by the loop condition itself.

        if (stop_training_flag) {
            break; // Exit epoch loop
        }
    } // End of epoch loop

    m_sycl_queue.wait_and_throw(); // Final wait before freeing resources

    sycl::free(full_input_samples_device, m_sycl_queue);
    sycl::free(full_target_samples_device, m_sycl_queue);
    deallocate_batch_temporaries_();

    m_EpochTotalLossHostHistory.resize(actual_epochs_run); // Adjust to actual number of epochs run
    return m_EpochTotalLossHostHistory;
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------
template <typename T> std::vector<T> Network<T>::predict(const std::vector<T> &input_samples, unsigned num_samples) {
    if (m_layer_configs.empty())
        throw std::runtime_error("Network not initialized or has no layers.");
    if (num_samples == 0)
        return {}; // Or throw error
    std::size_t num_input_features = m_layer_configs.front().neurons;
    if (input_samples.size() != num_samples * num_input_features)
        throw std::runtime_error("Prediction input size mismatch.");

    // Use allocate_batch_temporaries_ for m_LayerOutputActivations, m_LayerNetInputs
    // We only need one "epoch" for error storage, though it won't be used for loss.
    allocate_batch_temporaries_(num_samples, 1);

    // Copy input data to the first layer's output activation buffer
    // This can be done via a host pointer if USM shared, or explicit copy.
    // m_LayerOutputActivations[0].ptr is USM shared.
    std::copy_n(input_samples.data(), input_samples.size(), m_LayerOutputActivations[0].ptr);
    m_sycl_queue.wait_and_throw(); // Ensure copy is done if it involved device transfer implicitly

    sycl::event forward_pass_event; // Start with default event
    for (std::size_t l = 0; l < m_layer_configs.size() - 1; ++l) {
        forward_pass_event = forward_propagate_layer_(l, num_samples, forward_pass_event);
    }
    forward_pass_event.wait_and_throw(); // Wait for the entire forward pass to complete

    std::size_t num_output_features = m_layer_configs.back().neurons;
    std::vector<T> predictions(num_samples * num_output_features);
    // Copy results from the last layer's output activation buffer
    std::copy_n(m_LayerOutputActivations.back().ptr, predictions.size(), predictions.data());
    m_sycl_queue.wait_and_throw(); // Ensure copy is done

    deallocate_batch_temporaries_();
    return predictions;
}

// ---------------------------------------------------------------------------
// Weights and Biases Access
// ---------------------------------------------------------------------------
template <typename T>
auto Network<T>::weights_biases()
    -> std::pair<std::vector<std::vector<std::vector<T>>>, std::vector<std::vector<std::vector<T>>>> {
    // This method implies host access, ensure data is synced if needed (snapshot_history already does)
    return {m_WeightsHistory, m_BiasesHistory};
}

template <typename T>
void Network<T>::set_weights_biases(const std::vector<std::vector<T>> &new_weights,
                                    const std::vector<std::vector<T>> &new_biases) {
    if (new_weights.size() != m_Weights.size() || new_biases.size() != m_Biases.size())
        throw std::runtime_error("Layer count mismatch when setting weights/biases.");

    for (std::size_t l = 0; l < new_weights.size(); ++l) {
        if (new_weights[l].size() != m_Weights[l].rows * m_Weights[l].cols)
            throw std::runtime_error("Weight dimensions mismatch for layer " + std::to_string(l));
        if (new_biases[l].size() != m_Biases[l].elements)
            throw std::runtime_error("Bias dimensions mismatch for layer " + std::to_string(l));

        std::copy_n(new_weights[l].data(), new_weights[l].size(), m_Weights[l].ptr);
        std::copy_n(new_biases[l].data(), new_biases[l].size(), m_Biases[l].ptr);
    }
    m_sycl_queue.wait_and_throw(); // Ensure copies to device are complete

    // Clear history and add the new state as the initial state
    m_WeightsHistory.clear();
    m_BiasesHistory.clear();
    snapshot_history_();
}
