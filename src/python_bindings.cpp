// ============================================================================
//  syclnn_python.cpp   – pybind11 bindings for the event-driven USM Network
// ============================================================================
//
//  *   All enums are registered **before** the classes that use them.
//  *   Constructors now mirror the new C++ overload set.
//  *   train / predict wrappers expose the extra n_samples argument required
//      by the new implementation.
//  *   NumPy arrays are accepted everywhere for efficiency (C-contiguous
//      enforced via py::array::c_style | py::array::forcecast).
//  *   Loss history and weight/bias snapshots are returned as NumPy arrays /
//      nested Python lists, so the public Python API remains ergonomic.
//  *   The file is fully header-only – just place it next to `network.hpp`,
//      add it to your CMake / setup.py sources and rebuild.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

#include "network.hpp"

namespace py = pybind11;

// ============================================================================
//  Helper – copy NumPy data into std::vector<T>
// ============================================================================
template <typename T>
static std::vector<T> ndarray_to_vector(const py::array_t<T, py::array::c_style | py::array::forcecast> &arr) {
    const auto *ptr = static_cast<const T *>(arr.data());
    return std::vector<T>(ptr, ptr + arr.size());
}

// ============================================================================
//  Template helpers to register value-type wrappers
// ============================================================================
template <typename T> void declare_regularization(py::module_ &m, const std::string &suffix) {
    using Reg = Regularization<T>;

    // enum first
    py::enum_<typename Reg::Type>(m, ("RegularizationType_" + suffix).c_str())
        .value("Disabled", Reg::Type::Disabled)
        .value("L1", Reg::Type::L1)
        .value("L2", Reg::Type::L2)
        .value("ElasticNet", Reg::Type::ElasticNet)
        .export_values();

    // class
    py::class_<Reg>(m, ("Regularization_" + suffix).c_str())
        .def(py::init<typename Reg::Type, T, T>(), py::arg("type") = Reg::Type::Disabled,
             py::arg("lambda1") = static_cast<T>(0), py::arg("lambda2") = static_cast<T>(0))
        .def_readwrite("type", &Reg::type)
        .def_readwrite("lambda1", &Reg::lambda1)
        .def_readwrite("lambda2", &Reg::lambda2);
}

template <typename T> void declare_stop_criteria(py::module_ &m, const std::string &suffix) {
    using SC = StopCriteria<T>;

    py::enum_<typename SC::Type>(m, ("StopCriteriaType_" + suffix).c_str())
        .value("MaxEpochs", SC::Type::MaxEpochs)
        .value("MinError", SC::Type::MinError)
        .value("MinErrorChange", SC::Type::MinErrorChange)
        .export_values();

    py::class_<SC>(m, ("StopCriteria_" + suffix).c_str())
        .def(py::init<typename SC::Type, T>(), py::arg("type") = SC::Type::MaxEpochs,
             py::arg("threshold") = static_cast<T>(1e-4))
        .def_readwrite("type", &SC::type)
        .def_readwrite("threshold", &SC::threshold);
}

template <typename T> void declare_momentum(py::module_ &m, const std::string &suffix) {
    using Mom = MomentumConfig<T>;

    py::enum_<typename Mom::Type>(m, ("MomentumType_" + suffix).c_str())
        .value("Disabled", Mom::Type::Disabled)
        .value("Classical", Mom::Type::Classical)
        .export_values();

    py::class_<Mom>(m, ("MomentumConfig_" + suffix).c_str())
        .def(py::init<typename Mom::Type, T>(), py::arg("type") = Mom::Type::Disabled,
             py::arg("momentum_rate") = static_cast<T>(0))
        .def_readwrite("type", &Mom::type)
        .def_readwrite("momentum_rate", &Mom::momentum_rate);
}

template <typename T> void declare_adaptive_lr(py::module_ &m, const std::string &suffix) {
    using ALR = AdaptiveLearningRate<T>;

    py::enum_<typename ALR::Strategy>(m, ("AdaptiveLearningStrategy_" + suffix).c_str())
        .value("Constant", ALR::Strategy::Constant)
        .value("LinearDecay", ALR::Strategy::LinearDecay)
        .value("AdaGrad", ALR::Strategy::AdaGrad)
        .value("RMSProp", ALR::Strategy::RMSProp)
        .value("Adam", ALR::Strategy::Adam)
        .export_values();

    py::class_<ALR>(m, ("AdaptiveLearningRate_" + suffix).c_str())
        .def(py::init<typename ALR::Strategy, T, T, T, T>(), // Added T for beta2
             py::arg("strategy") = ALR::Strategy::Constant, py::arg("epsilon") = static_cast<T>(1e-8),
             py::arg("beta1") = static_cast<T>(0.9),   // Changed from rho
             py::arg("beta2") = static_cast<T>(0.999), // New argument
             py::arg("final_learning_rate") = static_cast<T>(1e-4))
        .def_readwrite("strategy", &ALR::strategy)
        .def_readwrite("epsilon", &ALR::epsilon)
        .def_readwrite("beta1", &ALR::beta1) // Changed from rho
        .def_readwrite("beta2", &ALR::beta2) // New member
        .def_readwrite("final_learning_rate", &ALR::final_lr);
}

// LayerDescription (non-templated)
static void declare_layer_description(py::module_ &m) {
    py::class_<LayerDescription>(m, "LayerDescription")
        .def(py::init<unsigned, ActivationType>(), py::arg("neurons"), py::arg("activation"))
        .def_readwrite("neurons", &LayerDescription::neurons)
        .def_readwrite("activation", &LayerDescription::activation);
}

// ============================================================================
//  Network<T> binding
// ============================================================================
template <typename T> void declare_network(py::module_ &m, const std::string &suffix) {
    using Net = Network<T>;
    const std::string name = "Network_" + suffix;

    py::class_<Net>(m, name.c_str())
        // --- main constructor (random init, seed from time) -----------------
        .def(py::init<std::vector<LayerDescription>, T, Regularization<T>, BackPropagation, AdaptiveLearningRate<T>,
                      StopCriteria<T>, MomentumConfig<T>>(),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization"),
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{},
             py::arg("stop_criteria") = StopCriteria<T>{}, py::arg("momentum") = MomentumConfig<T>{})
        // --- constructor with explicit RNG seed -----------------------------
        .def(py::init<std::vector<LayerDescription>, T, Regularization<T>, BackPropagation, AdaptiveLearningRate<T>,
                      StopCriteria<T>, MomentumConfig<T>, unsigned>(),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization"),
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{},
             py::arg("stop_criteria") = StopCriteria<T>{}, py::arg("momentum") = MomentumConfig<T>{}, py::arg("seed"))
        // --- constructor with user-supplied weights / biases -----------------
        .def(py::init<std::vector<LayerDescription>, T, Regularization<T>, BackPropagation, AdaptiveLearningRate<T>,
                      StopCriteria<T>, MomentumConfig<T>,
                      std::vector<std::vector<T>> &, // weights
                      std::vector<std::vector<T>> &  // biases
                      >(),
             py::arg("layers"), py::arg("learning_rate"), py::arg("regularization"),
             py::arg("backpropagation") = BackPropagation::Standard,
             py::arg("adaptive_learning_rate") = AdaptiveLearningRate<T>{},
             py::arg("stop_criteria") = StopCriteria<T>{}, py::arg("momentum") = MomentumConfig<T>{},
             py::arg("initial_weights"), py::arg("initial_biases"))
        // --------------------------------------------------------------------
        //  train
        // --------------------------------------------------------------------
        .def(
            "train",
            [](Net &self, py::array_t<T, py::array::c_style | py::array::forcecast> X,
               py::array_t<T, py::array::c_style | py::array::forcecast> Y, unsigned n_samples, unsigned batch_size,
               unsigned max_epochs) {
                auto x_vec = ndarray_to_vector(X);
                auto y_vec = ndarray_to_vector(Y);
                std::vector<T> loss = self.train(x_vec, y_vec, n_samples, batch_size, max_epochs);
                // return as NumPy 1-D array
                return py::array_t<T>(py::cast(loss));
            },
            py::arg("x"), py::arg("y"), py::arg("n_samples"), py::arg("batch_size"), py::arg("max_epochs"),
            R"pbdoc(
                 Train the network.

                 Parameters
                 ----------
                 x : ndarray[float|double]
                     Flattened training inputs.
                 y : ndarray[float|double]
                     Flattened target outputs.
                 n_samples : int
                     Number of training patterns in the flattened arrays.
                 batch_size : int
                     Mini-batch size.
                 max_epochs : int
                     Maximum number of epochs.

                 Returns
                 -------
                 ndarray
                     Loss value at every finished epoch.
             )pbdoc")
        // --------------------------------------------------------------------
        //  predict
        // --------------------------------------------------------------------
        .def(
            "predict",
            [](Net &self, py::array_t<T, py::array::c_style | py::array::forcecast> X, unsigned n_samples) {
                auto x_vec = ndarray_to_vector(X);
                std::vector<T> y_vec = self.predict(x_vec, n_samples);
                return py::array_t<T>(py::cast(y_vec));
            },
            py::arg("x"), py::arg("n_samples"),
            R"pbdoc(
                 Forward-propagate unseen data.

                 Parameters
                 ----------
                 x : ndarray[float|double]
                     Flattened input patterns.
                 n_samples : int
                     Number of patterns in *x*.

                 Returns
                 -------
                 ndarray
                     Flattened network output.
             )pbdoc")
        // --------------------------------------------------------------------
        //  weights / biases history as Python object
        // --------------------------------------------------------------------
        .def_property_readonly(
            "weights_biases",
            [](Net &self) {
                using Hist =
                    std::pair<std::vector<std::vector<std::vector<T>>>, std::vector<std::vector<std::vector<T>>>>;
                Hist hist = self.weights_biases();
                return py::cast(std::move(hist));
            },
            R"pbdoc(
                 Complete history of weights and biases captured after each
                 finished epoch (epoch 0 is the initial state).

                 Returns
                 -------
                 Tuple[List[List[List]], List[List[List]]]
                     (weights_history, biases_history)
             )pbdoc")
        // --------------------------------------------------------------------
        //  overwrite weights / biases ----------------------------------------
        .def("set_weights_biases", &Net::set_weights_biases, py::arg("weights"), py::arg("biases"),
             R"pbdoc(
                 Replace all weights / biases and reset the internal history.

                 Parameters
                 ----------
                 weights : List[List[float|double]]
                     Flattened weight matrices, one entry per layer.
                 biases : List[List[float|double]]
                     Bias vectors, one entry per layer.
             )pbdoc");
}

// ============================================================================
//  PYBIND11 MODULE
// ============================================================================
PYBIND11_MODULE(_syclnn, m) {
    m.doc() = "SYCL neural-network bindings (event-driven USM version)";

    // ----------------- standalone enums ------------------------------------
    py::enum_<ActivationType>(m, "ActivationType")
        .value("Disabled", ActivationType::Disabled)
        .value("Sigmoid", ActivationType::Sigmoid)
        .value("Tanh", ActivationType::Tanh)
        .value("ReLU", ActivationType::ReLU)
        .value("LeakyReLU", ActivationType::LeakyReLU)
        .value("ELU", ActivationType::ELU)
        .export_values();

    py::enum_<BackPropagation>(m, "BackPropagation").value("Standard", BackPropagation::Standard).export_values();

    // ----------------- POD helper struct -----------------------------------
    declare_layer_description(m);

    // ----------------- float / double template specialisations -------------
    declare_regularization<float>(m, "float");
    declare_stop_criteria<float>(m, "float");
    declare_momentum<float>(m, "float");
    declare_adaptive_lr<float>(m, "float");
    declare_network<float>(m, "float");

    declare_regularization<double>(m, "double");
    declare_stop_criteria<double>(m, "double");
    declare_momentum<double>(m, "double");
    declare_adaptive_lr<double>(m, "double");
    declare_network<double>(m, "double");

    // ----------------- utility – query devices ------------------------------
    m.def(
        "get_sycl_devices",
        [] {
            std::vector<sycl::device> devs = sycl::device::get_devices();
            std::vector<std::string> names;
            names.reserve(devs.size());
            for (const auto &d : devs)
                names.emplace_back(d.get_info<sycl::info::device::name>());
            return names;
        },
        "Return a list with the names of all visible SYCL devices.");
}
