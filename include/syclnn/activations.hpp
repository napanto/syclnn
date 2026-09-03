// SPDX-License-Identifier: LGPL-3.0-only
// syclnn - activation functions and derivatives, callable from SYCL kernels.
//
// f(net) and f'(net) follow the lecture definitions; f' can also be evaluated
// from the stored output o = f(net) for the activations where that is cheaper
// (sigma' = o(1-o), tanh' = 1-o^2, elu' = o + alpha for net <= 0), which is what
// Options::derivative_from_output selects.
#pragma once

#include <sycl/sycl.hpp>

#include "syclnn/config.hpp"

namespace syclnn {

template <typename T> inline constexpr T leaky_relu_alpha = T(0.01);
template <typename T> inline constexpr T elu_alpha = T(1);

/// sgn(x) in {-1, 0, 1}
template <typename T> inline constexpr T sgn(T x) noexcept { return T(0) < x ? T(1) : (x < T(0) ? T(-1) : T(0)); }

/// Compile-time activation dispatch: Act<A>::f / ::df / ::df_out.
template <ActivationType A> struct Act;

template <> struct Act<ActivationType::Disabled> {
    template <typename T> static inline T f(T x) { return x; }
    template <typename T> static inline T df(T) { return T(1); }
    template <typename T> static inline T df_out(T, T) { return T(1); }
};
template <> struct Act<ActivationType::Sigmoid> {
    template <typename T> static inline T f(T x) { return T(1) / (T(1) + sycl::exp(-x)); }
    template <typename T> static inline T df(T x) {
        T s = f(x);
        return s * (T(1) - s);
    }
    template <typename T> static inline T df_out(T o, T) { return o * (T(1) - o); }
};
template <> struct Act<ActivationType::Tanh> {
    template <typename T> static inline T f(T x) { return sycl::tanh(x); }
    template <typename T> static inline T df(T x) {
        T t = sycl::tanh(x);
        return T(1) - t * t;
    }
    template <typename T> static inline T df_out(T o, T) { return T(1) - o * o; }
};
template <> struct Act<ActivationType::ReLU> {
    template <typename T> static inline T f(T x) { return sycl::fmax(T(0), x); }
    template <typename T> static inline T df(T x) { return x > T(0) ? T(1) : T(0); }
    template <typename T> static inline T df_out(T, T x) { return df(x); }
};
template <> struct Act<ActivationType::LeakyReLU> {
    template <typename T> static inline T f(T x) { return x > T(0) ? x : leaky_relu_alpha<T> * x; }
    template <typename T> static inline T df(T x) { return x > T(0) ? T(1) : leaky_relu_alpha<T>; }
    template <typename T> static inline T df_out(T, T x) { return df(x); }
};
template <> struct Act<ActivationType::ELU> {
    template <typename T> static inline T f(T x) { return x > T(0) ? x : elu_alpha<T> * (sycl::exp(x) - T(1)); }
    template <typename T> static inline T df(T x) { return x > T(0) ? T(1) : elu_alpha<T> * sycl::exp(x); }
    template <typename T> static inline T df_out(T o, T x) { return x > T(0) ? T(1) : o + elu_alpha<T>; }
};

/// Run-time activation dispatch (the 0.1 "switch inside the kernel" path).
template <typename T> inline T activate(ActivationType a, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::f(x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::f(x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::f(x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::f(x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::f(x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::f(x);
    }
    return x;
}
template <typename T> inline T derivative(ActivationType a, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::df(x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::df(x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::df(x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::df(x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::df(x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::df(x);
    }
    return T(1);
}
template <typename T> inline T derivative_from_output(ActivationType a, T o, T x) {
    switch (a) {
    case ActivationType::Disabled: return Act<ActivationType::Disabled>::df_out(o, x);
    case ActivationType::Sigmoid: return Act<ActivationType::Sigmoid>::df_out(o, x);
    case ActivationType::Tanh: return Act<ActivationType::Tanh>::df_out(o, x);
    case ActivationType::ReLU: return Act<ActivationType::ReLU>::df_out(o, x);
    case ActivationType::LeakyReLU: return Act<ActivationType::LeakyReLU>::df_out(o, x);
    case ActivationType::ELU: return Act<ActivationType::ELU>::df_out(o, x);
    }
    return T(1);
}

/// Call `fn(std::integral_constant<ActivationType, A>{})` for the run-time value `a`
/// so that kernels can be instantiated per activation (Options::specialized_kernels).
template <typename Fn> inline decltype(auto) dispatch_activation(ActivationType a, Fn &&fn) {
    switch (a) {
    case ActivationType::Sigmoid: return fn(std::integral_constant<ActivationType, ActivationType::Sigmoid>{});
    case ActivationType::Tanh: return fn(std::integral_constant<ActivationType, ActivationType::Tanh>{});
    case ActivationType::ReLU: return fn(std::integral_constant<ActivationType, ActivationType::ReLU>{});
    case ActivationType::LeakyReLU: return fn(std::integral_constant<ActivationType, ActivationType::LeakyReLU>{});
    case ActivationType::ELU: return fn(std::integral_constant<ActivationType, ActivationType::ELU>{});
    case ActivationType::Disabled:
    default: return fn(std::integral_constant<ActivationType, ActivationType::Disabled>{});
    }
}

} // namespace syclnn
