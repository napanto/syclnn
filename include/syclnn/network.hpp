// SPDX-License-Identifier: LGPL-3.0-only
// syclnn - SYCL-accelerated feed-forward neural network library
// Copyright (C) 2025-2026 Antonio Napolitano, Suqi Chen
//
// network.hpp: Network<T>, a multi-layer perceptron trained with mini-batch
// back-propagation on any SYCL device.  GEMM/GEMV/asum/nrm2 go to oneMath;
// everything else is a hand-written SYCL kernel.  Tensors are column-major USM
// allocations (one sample per column, neuron index fastest).  All device work is
// asynchronous and ordered through SYCL events; the host synchronises once per
// epoch to read the loss.  Every optimisation introduced in 0.2 can be switched
// back to the 0.1 behaviour through Options (see config.hpp).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <ctime>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "syclnn/activations.hpp"
#include "syclnn/blas.hpp"
#include "syclnn/config.hpp"
#include "syclnn/device.hpp"
#include "syclnn/profile.hpp"

namespace syclnn {

namespace detail {

template <typename T> inline T *usm_alloc(std::size_t n, sycl::queue &q, MemoryKind kind) {
    if (n == 0)
        return nullptr;
    T *p = nullptr;
    switch (kind) {
    case MemoryKind::Device: p = sycl::malloc_device<T>(n, q); break;
    case MemoryKind::Shared: p = sycl::malloc_shared<T>(n, q); break;
    case MemoryKind::Host: p = sycl::malloc_host<T>(n, q); break;
    }
    if (!p)
        throw std::runtime_error("syclnn: USM allocation of " + std::to_string(n * sizeof(T)) + " bytes failed");
    return p;
}

/// RAII USM allocation bound to a queue (move-only).
template <typename T> class UsmBuffer {
  public:
    UsmBuffer() = default;
    UsmBuffer(std::size_t n, sycl::queue &q, MemoryKind kind) : m_ctx(q.get_context()), m_n(n), m_ptr(usm_alloc<T>(n, q, kind)) {}
    UsmBuffer(const UsmBuffer &) = delete;
    UsmBuffer &operator=(const UsmBuffer &) = delete;
    UsmBuffer(UsmBuffer &&o) noexcept : m_ctx(std::move(o.m_ctx)), m_n(o.m_n), m_ptr(o.m_ptr) {
        o.m_ptr = nullptr;
        o.m_n = 0;
    }
    UsmBuffer &operator=(UsmBuffer &&o) noexcept {
        if (this != &o) {
            release();
            m_ctx = std::move(o.m_ctx);
            m_n = o.m_n;
            m_ptr = o.m_ptr;
            o.m_ptr = nullptr;
            o.m_n = 0;
        }
        return *this;
    }
    ~UsmBuffer() { release(); }
    void release() noexcept {
        if (m_ptr && m_ctx) {
            try {
                sycl::free(m_ptr, *m_ctx);
            } catch (...) {
            }
            m_ptr = nullptr;
            m_n = 0;
        }
    }
    T *data() const { return m_ptr; }
    std::size_t size() const { return m_n; }
    explicit operator bool() const { return m_ptr != nullptr; }
    std::size_t bytes() const { return m_n * sizeof(T); }

  private:
    std::optional<sycl::context> m_ctx;
    std::size_t m_n = 0;
    T *m_ptr = nullptr;
};

inline std::uint32_t next_mt(std::mt19937 &g) { return static_cast<std::uint32_t>(g()); }

/// Fisher-Yates with j = i + mt() % (n - i): reproduced bit-exactly by the testkit.
inline std::vector<std::uint32_t> shuffle_permutation(std::uint32_t n, std::mt19937 &g) {
    std::vector<std::uint32_t> perm(n);
    for (std::uint32_t i = 0; i < n; ++i)
        perm[i] = i;
    for (std::uint32_t i = 0; i + 1 < n; ++i) {
        std::uint32_t j = i + next_mt(g) % (n - i);
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

} // namespace detail

/**
 * @tparam T float or double
 */
template <typename T> class Network {
    static_assert(std::is_floating_point_v<T>, "Network<T> requires float or double");

  public:
    using history_t = std::vector<std::vector<std::vector<T>>>;

    /// Explicit initial weights (column-major, (n_{l+1} x n_l)) and biases.
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
            const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
            Options options = Options{});
    /// Random initialisation in [-0.1, 0.1] from std::mt19937(seed) (seed 0 = time).
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
            Options options = Options{});
    /// Random initialisation seeded from the clock.
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options = Options{});
    ~Network() noexcept;
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;

    /**
     * Mini-batch back-propagation.
     * @param input_samples  num_samples x n_in values, sample-major (row-major NumPy (N, n_in))
     * @param target_samples num_samples x n_out values
     * @return total loss (mean data loss + regularisation penalty) of every epoch run
     */
    std::vector<T> train(const std::vector<T> &input_samples, const std::vector<T> &target_samples, unsigned num_samples,
                         unsigned batch_size, unsigned max_epochs);
    /// Forward pass; batch_size 0 = the whole set in one batch.
    std::vector<T> predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size = 0);

    /// Snapshots {weights, biases}[snapshot][layer][flat column-major values].
    std::pair<history_t, history_t> weights_biases() const { return {m_weights_history, m_biases_history}; }
    std::vector<std::vector<T>> weights() const;
    std::vector<std::vector<T>> biases() const;
    void set_weights_biases(const std::vector<std::vector<T>> &new_weights, const std::vector<std::vector<T>> &new_biases);

    const Profile &profile() const { return m_prof.profile(); }
    void reset_profile() { m_prof.reset(); }
    const Options &options() const { return m_opts; }
    const std::vector<LayerDescription> &layers() const { return m_layers; }
    std::string device_name() const { return m_device.template get_info<sycl::info::device::name>(); }
    DeviceInfo device_info() const { return describe(m_device, -1); }
    std::string blas_backend() const { return blas_backend_name(m_blas.backend()); }
    std::size_t parameter_count() const;
    std::size_t num_layers() const { return m_layers.size(); }

  private:
    using Buf = detail::UsmBuffer<T>;
    using ev_list = std::vector<sycl::event>;

    // ---- configuration ----
    std::vector<LayerDescription> m_layers;
    T m_lr{};
    Regularization<T> m_reg;
    MomentumConfig<T> m_mom;
    BackPropagation m_bp;
    AdaptiveLearningRate<T> m_adapt;
    StopCriteria<T> m_stop;
    Options m_opts;
    MemoryKind m_kind;
    std::size_t m_L = 0; ///< number of weight layers = layers.size() - 1

    // ---- SYCL (initialised in the mem-init list: a default-constructed queue
    //      would already pick and initialise a device) ----
    sycl::device m_device;
    sycl::queue m_queue;
    Blas m_blas;
    Profiler m_prof;
    static sycl::queue make_queue(const sycl::device &dev, const Options &o);

    // ---- parameters and optimiser state (per weight layer) ----
    std::vector<Buf> m_W, m_b, m_mW, m_vW, m_mb, m_vb;

    // ---- workspace (capacity m_cap samples) ----
    std::size_t m_cap = 0;
    std::vector<Buf> m_act;   ///< act[l] (n_l x B), l = 0..L (act[0] only used without direct_input)
    std::vector<Buf> m_net;   ///< net[l] (n_{l+1} x B) = W_l act_l + b_l
    std::vector<Buf> m_delta; ///< delta[l] (n_{l+1} x B)
    std::vector<Buf> m_gW, m_gb;
    Buf m_ones;                              ///< B ones for the bias GEMV
    detail::UsmBuffer<double> m_loss_acc_d;  ///< [0] data loss, [1] regularisation penalty
    Buf m_loss_acc_t;                        ///< 0.1-style loss accumulator of type T
    Buf m_reg_tmp;                           ///< 2 scalars per layer: asum, nrm2
    detail::UsmBuffer<double> m_host_scalars; ///< pinned host readback: loss, penalty
    Buf m_host_scalar_t;

    // ---- bookkeeping ----
    unsigned m_adam_step = 0;
    history_t m_weights_history, m_biases_history;

    // ---- helpers ----
    void validate_device();
    void init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed);
    void ensure_workspace(std::size_t batch);
    void release_workspace();
    void snapshot_history();
    void h2d(T *dst, const T *src, std::size_t n, const ev_list &deps, ev_list *out = nullptr);
    void d2h_wait(T *dst, const T *src, std::size_t n);
    ev_list join(const ev_list &evs);
    sycl::event ev_join(const ev_list &evs);
    void wait_all();

    template <typename F> sycl::event launch(std::size_t n, const ev_list &deps, Phase phase, F fn);

    sycl::event forward_layer(std::size_t l, const T *in, std::size_t B, const ev_list &deps, sycl::event *gemm_ev);
    sycl::event output_delta_loss(const T *targets, std::size_t B, const ev_list &deps);
    sycl::event hidden_delta(std::size_t l, std::size_t B, const ev_list &deps);
    void gradients(std::size_t l, const T *in, std::size_t B, const ev_list &deps, sycl::event &ev_w, sycl::event &ev_b);
    sycl::event update(std::size_t l, unsigned adam_step, unsigned epoch, unsigned max_epochs, const ev_list &deps);
    sycl::event penalty(const ev_list &deps);
};

/* ============================================================================
                                  IMPLEMENTATION
============================================================================ */

// ---------------------------------------------------------------- construction

template <typename T> sycl::queue Network<T>::make_queue(const sycl::device &dev, const Options &o) {
    auto handler = [](sycl::exception_list el) {
        for (auto &e : el)
            std::rethrow_exception(e);
    };
    const bool in_order = o.queue == QueueOrder::InOrder;
    if (o.profile && in_order)
        return sycl::queue(dev, handler, {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}});
    if (o.profile)
        return sycl::queue(dev, handler, {sycl::property::queue::enable_profiling{}});
    if (in_order)
        return sycl::queue(dev, handler, {sycl::property::queue::in_order{}});
    return sycl::queue(dev, handler);
}

template <typename T> void Network<T>::validate_device() {
    m_kind = m_opts.memory;
    if (m_kind == MemoryKind::Shared && !m_device.has(sycl::aspect::usm_shared_allocations))
        throw std::invalid_argument("syclnn: device '" + device_name() + "' does not support USM shared allocations");
    if (m_kind == MemoryKind::Host && !m_device.has(sycl::aspect::usm_host_allocations))
        throw std::invalid_argument("syclnn: device '" + device_name() + "' does not support USM host allocations");
    // the loss/penalty accumulators are double for every T
    if (!m_device.has(sycl::aspect::fp64))
        throw std::invalid_argument("syclnn: device '" + device_name() + "' has no fp64 support (needed by the loss accumulators)");
    if (!m_device.has(sycl::aspect::usm_host_allocations))
        throw std::invalid_argument("syclnn: device '" + device_name() + "' does not support USM host allocations (used for staging)");
    if (std::is_same_v<T, double> && !m_opts.loss_reduction && !m_device.has(sycl::aspect::atomic64))
        throw std::invalid_argument("syclnn: device '" + device_name() + "' has no 64-bit atomics (loss_reduction=false needs them)");
    m_blas = Blas(parse_blas_backend(m_opts.blas));
    m_blas.check(m_device);
    m_prof = Profiler(m_opts.profile);
    m_host_scalars = detail::UsmBuffer<double>(2, m_queue, MemoryKind::Host);
    m_host_scalar_t = Buf(1, m_queue, MemoryKind::Host);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
                    const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)), m_device(select_device(m_opts.device)), m_queue(make_queue(m_device, m_opts)) {
    validate_device();
    init_parameters(&initial_weights, &initial_biases, 0);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)), m_device(select_device(m_opts.device)), m_queue(make_queue(m_device, m_opts)) {
    validate_device();
    init_parameters(nullptr, nullptr, seed);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options)
    : Network(std::move(layers), learning_rate, reg, bp, adapt, stop, mom, static_cast<unsigned>(std::time(nullptr)),
              std::move(options)) {}

template <typename T> Network<T>::~Network() noexcept {
    try {
        m_queue.wait_and_throw();
    } catch (...) {
        // an asynchronous error must not terminate the process from a destructor
    }
    // UsmBuffer destructors release everything
}

template <typename T>
void Network<T>::init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed) {
    // ---- validate everything before a single byte is submitted to the device ----
    if (m_layers.size() < 2)
        throw std::invalid_argument("syclnn: a network needs at least an input and an output layer");
    for (const auto &l : m_layers)
        if (l.neurons == 0)
            throw std::invalid_argument("syclnn: layer neuron count must be greater than 0");
    m_L = m_layers.size() - 1;
    const bool random_init = (w == nullptr || b == nullptr);
    if (!random_init) {
        if (w->size() != m_L)
            throw std::invalid_argument("syclnn: initial weights have " + std::to_string(w->size()) + " layers, expected " +
                                        std::to_string(m_L));
        if (b->size() != m_L)
            throw std::invalid_argument("syclnn: initial biases have " + std::to_string(b->size()) + " layers, expected " +
                                        std::to_string(m_L));
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
            const std::size_t n_out = m_layers[l + 1].neurons;
            if ((*w)[l].size() != nw)
                throw std::invalid_argument("syclnn: initial weights of layer " + std::to_string(l) + " have " +
                                            std::to_string((*w)[l].size()) + " values, expected " + std::to_string(nw));
            if ((*b)[l].size() != n_out)
                throw std::invalid_argument("syclnn: initial biases of layer " + std::to_string(l) + " have " +
                                            std::to_string((*b)[l].size()) + " values, expected " + std::to_string(n_out));
        }
    }
    std::mt19937 rng(seed == 0 ? static_cast<unsigned>(std::time(nullptr)) : seed);
    std::uniform_real_distribution<T> dist(T(-0.1), T(0.1));
    std::vector<std::vector<T>> host_w(m_L), host_b(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
        const std::size_t n_out = m_layers[l + 1].neurons;
        if (random_init) {
            host_w[l].resize(nw);
            host_b[l].resize(n_out);
            for (auto &x : host_w[l])
                x = dist(rng);
            for (auto &x : host_b[l])
                x = dist(rng);
        } else {
            host_w[l] = (*w)[l];
            host_b[l] = (*b)[l];
        }
    }

    // ---- allocate and upload; if anything throws, drain the queue first so the
    //      buffers (members, freed after this frame unwinds) are no longer in use ----
    struct Quiesce {
        sycl::queue &q;
        ~Quiesce() {
            if (std::uncaught_exceptions() > 0) {
                try {
                    q.wait();
                } catch (...) {
                }
            }
        }
    } quiesce{m_queue};
    m_W.clear();
    m_b.clear();
    m_mW.clear();
    m_vW.clear();
    m_mb.clear();
    m_vb.clear();
    ev_list evs;
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::size_t nw = host_w[l].size(), n_out = host_b[l].size();
        m_W.emplace_back(nw, m_queue, m_kind);
        m_b.emplace_back(n_out, m_queue, m_kind);
        m_mW.emplace_back(nw, m_queue, m_kind);
        m_vW.emplace_back(nw, m_queue, m_kind);
        m_mb.emplace_back(n_out, m_queue, m_kind);
        m_vb.emplace_back(n_out, m_queue, m_kind);
        h2d(m_W[l].data(), host_w[l].data(), nw, {}, &evs);
        h2d(m_b[l].data(), host_b[l].data(), n_out, {}, &evs);
        for (Buf *buf : {&m_mW[l], &m_vW[l]}) {
            auto e = m_queue.fill(buf->data(), T(0), nw);
            m_prof.record(Phase::Other, e);
            evs.push_back(e);
        }
        for (Buf *buf : {&m_mb[l], &m_vb[l]}) {
            auto e = m_queue.fill(buf->data(), T(0), n_out);
            m_prof.record(Phase::Other, e);
            evs.push_back(e);
        }
    }
    m_prof.timed_wait([&] { sycl::event::wait_and_throw(evs); });
    m_prof.flush();
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(std::move(host_w));
    m_biases_history.push_back(std::move(host_b));
}

template <typename T> std::size_t Network<T>::parameter_count() const {
    std::size_t n = 0;
    for (std::size_t l = 0; l < m_L; ++l)
        n += std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons + m_layers[l + 1].neurons;
    return n;
}

// ---------------------------------------------------------------- memory helpers

template <typename T> void Network<T>::h2d(T *dst, const T *src, std::size_t n, const ev_list &deps, ev_list *out) {
    if (n == 0)
        return;
    auto e = m_queue.memcpy(dst, src, n * sizeof(T), deps);
    m_prof.record(Phase::H2D, e, n * sizeof(T));
    if (out)
        out->push_back(e);
}

template <typename T> void Network<T>::d2h_wait(T *dst, const T *src, std::size_t n) {
    if (n == 0)
        return;
    auto e = m_queue.memcpy(dst, src, n * sizeof(T));
    m_prof.record(Phase::D2H, e, n * sizeof(T));
    m_prof.timed_wait([&] { e.wait_and_throw(); });
}

template <typename T> void Network<T>::wait_all() {
    m_prof.timed_wait([&] { m_queue.wait_and_throw(); });
    m_prof.flush();
}

template <typename T> sycl::event Network<T>::ev_join(const ev_list &evs) {
    // the 0.1 way of merging dependencies: an empty kernel
    auto e = m_queue.submit([&](sycl::handler &h) {
        h.depends_on(evs);
        h.single_task([] {});
    });
    m_prof.record(Phase::Other, e);
    return e;
}

template <typename T> typename Network<T>::ev_list Network<T>::join(const ev_list &evs) {
    if (m_opts.join_kernels)
        return {ev_join(evs)};
    return evs;
}

template <typename T> void Network<T>::ensure_workspace(std::size_t batch) {
    if (batch <= m_cap && !m_act.empty())
        return;
    release_workspace();
    try {
        m_act.reserve(m_L + 1);
        for (std::size_t l = 0; l <= m_L; ++l) {
            const bool needed = (l > 0) || !m_opts.direct_input;
            m_act.emplace_back(needed ? std::size_t(m_layers[l].neurons) * batch : 0, m_queue, m_kind);
        }
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t n_in = m_layers[l].neurons, n_out = m_layers[l + 1].neurons;
            m_net.emplace_back(n_out * batch, m_queue, m_kind);
            m_delta.emplace_back(n_out * batch, m_queue, m_kind);
            m_gW.emplace_back(n_out * n_in, m_queue, m_kind);
            m_gb.emplace_back(n_out, m_queue, m_kind);
        }
        m_ones = Buf(batch, m_queue, m_kind);
        m_loss_acc_d = detail::UsmBuffer<double>(2, m_queue, m_kind);
        m_loss_acc_t = Buf(1, m_queue, m_kind);
        m_reg_tmp = Buf(2 * m_L, m_queue, m_kind);
        ev_list evs;
        evs.push_back(m_queue.fill(m_ones.data(), T(1), batch));
        evs.push_back(m_queue.fill(m_loss_acc_d.data(), 0.0, 2));
        evs.push_back(m_queue.fill(m_loss_acc_t.data(), T(0), 1));
        for (auto &e : evs)
            m_prof.record(Phase::Other, e);
        m_prof.timed_wait([&] { sycl::event::wait_and_throw(evs); });
    } catch (...) {
        try {
            m_queue.wait();
        } catch (...) {
        }
        release_workspace();
        throw;
    }
    m_cap = batch;
}

template <typename T> void Network<T>::release_workspace() {
    if (!m_act.empty() || m_ones)
        wait_all();
    m_act.clear();
    m_net.clear();
    m_delta.clear();
    m_gW.clear();
    m_gb.clear();
    m_ones.release();
    m_loss_acc_d.release();
    m_loss_acc_t.release();
    m_reg_tmp.release();
    m_cap = 0;
}

template <typename T> std::vector<std::vector<T>> Network<T>::weights() const {
    auto *self = const_cast<Network *>(this);
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_W[l].size());
        self->d2h_wait(out[l].data(), m_W[l].data(), m_W[l].size());
    }
    self->m_prof.flush();
    return out;
}

template <typename T> std::vector<std::vector<T>> Network<T>::biases() const {
    auto *self = const_cast<Network *>(this);
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_b[l].size());
        self->d2h_wait(out[l].data(), m_b[l].data(), m_b[l].size());
    }
    self->m_prof.flush();
    return out;
}

template <typename T> void Network<T>::snapshot_history() {
    m_weights_history.push_back(weights());
    m_biases_history.push_back(biases());
}

template <typename T>
void Network<T>::set_weights_biases(const std::vector<std::vector<T>> &new_weights,
                                    const std::vector<std::vector<T>> &new_biases) {
    if (new_weights.size() != m_L || new_biases.size() != m_L)
        throw std::invalid_argument("syclnn: layer count mismatch in set_weights_biases");
    for (std::size_t l = 0; l < m_L; ++l) {
        if (new_weights[l].size() != m_W[l].size())
            throw std::invalid_argument("syclnn: weight size mismatch for layer " + std::to_string(l));
        if (new_biases[l].size() != m_b[l].size())
            throw std::invalid_argument("syclnn: bias size mismatch for layer " + std::to_string(l));
    }
    wait_all();
    ev_list evs;
    for (std::size_t l = 0; l < m_L; ++l) {
        h2d(m_W[l].data(), new_weights[l].data(), m_W[l].size(), {}, &evs);
        h2d(m_b[l].data(), new_biases[l].data(), m_b[l].size(), {}, &evs);
    }
    m_prof.timed_wait([&] { sycl::event::wait_and_throw(evs); });
    m_prof.flush();
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(new_weights);
    m_biases_history.push_back(new_biases);
}

// ---------------------------------------------------------------- kernels

template <typename T>
template <typename F>
sycl::event Network<T>::launch(std::size_t n, const ev_list &deps, Phase phase, F fn) {
    sycl::event e;
    const std::size_t wg = m_opts.workgroup_size;
    if (wg == 0) {
        e = m_queue.submit([&](sycl::handler &h) {
            h.depends_on(deps);
            h.parallel_for(sycl::range<1>(n), [=](sycl::item<1> it) { fn(it.get_id(0)); });
        });
    } else {
        const std::size_t global = (n + wg - 1) / wg * wg;
        e = m_queue.submit([&](sycl::handler &h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
                const std::size_t i = it.get_global_id(0);
                if (i < n)
                    fn(i);
            });
        });
    }
    m_prof.record(phase, e);
    return e;
}

/// net_l = W_l * in + b_l ; act_{l+1} = f(net_l)
template <typename T>
sycl::event Network<T>::forward_layer(std::size_t l, const T *in, std::size_t B, const ev_list &deps, sycl::event *gemm_ev) {
    const std::size_t M = m_layers[l + 1].neurons, K = m_layers[l].neurons;
    T *net = m_net[l].data();
    T *out = m_act[l + 1].data();
    const T *bias = m_b[l].data();
    sycl::event ge = m_blas.gemm(m_queue, Blas::N, Blas::N, std::int64_t(M), std::int64_t(B), std::int64_t(K), T(1),
                                 m_W[l].data(), std::int64_t(M), in, std::int64_t(K), T(0), net, std::int64_t(M), deps);
    m_prof.record(Phase::Gemm, ge);
    if (gemm_ev)
        *gemm_ev = ge;
    const std::size_t n = M * B;
    const ActivationType act = m_layers[l + 1].activation;
    if (m_opts.specialized_kernels) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            return launch(n, {ge}, Phase::Act, [=](std::size_t i) {
                const T z = net[i] + bias[i % M];
                net[i] = z;
                out[i] = Act<A>::f(z);
            });
        });
    }
    return launch(n, {ge}, Phase::Act, [=](std::size_t i) {
        const T z = net[i] + bias[i % M];
        net[i] = z;
        out[i] = activate(act, z);
    });
}

/// delta_L = (t - o) * f'(net_L) and loss += 0.5 (t - o)^2
template <typename T>
sycl::event Network<T>::output_delta_loss(const T *targets, std::size_t B, const ev_list &deps) {
    const std::size_t l = m_L - 1;
    const std::size_t M = m_layers[m_L].neurons;
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[m_L].data();
    T *delta = m_delta[l].data();
    const ActivationType act = m_layers[m_L].activation;
    const bool from_out = m_opts.derivative_from_output;
    const bool spec = m_opts.specialized_kernels;
    const std::size_t wg = m_opts.workgroup_size;

    auto body = [=](std::size_t i, auto fprime) -> double {
        const T o = out[i];
        const T e = targets[i] - o;
        delta[i] = e * fprime(o, net[i]);
        const double ed = static_cast<double>(e);
        return 0.5 * ed * ed;
    };
    auto submit = [&](auto fprime) {
        sycl::event e;
        if (m_opts.loss_reduction) {
            double *acc = m_loss_acc_d.data();
            if (wg == 0) {
                e = m_queue.submit([&](sycl::handler &h) {
                    h.depends_on(deps);
                    h.parallel_for(sycl::range<1>(n), sycl::reduction(acc, sycl::plus<double>()),
                                   [=](sycl::item<1> it, auto &sum) { sum += body(it.get_id(0), fprime); });
                });
            } else {
                const std::size_t global = (n + wg - 1) / wg * wg;
                e = m_queue.submit([&](sycl::handler &h) {
                    h.depends_on(deps);
                    h.parallel_for(sycl::nd_range<1>(global, wg), sycl::reduction(acc, sycl::plus<double>()),
                                   [=](sycl::nd_item<1> it, auto &sum) {
                                       const std::size_t i = it.get_global_id(0);
                                       if (i < n)
                                           sum += body(i, fprime);
                                   });
                });
            }
            m_prof.record(Phase::Loss, e);
            return e;
        }
        // 0.1: every work-item does an atomic add on one scalar of type T
        T *acc = m_loss_acc_t.data();
        return launch(n, deps, Phase::Loss, [=](std::size_t i) {
            const T contrib = static_cast<T>(body(i, fprime));
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(acc[0]);
            a.fetch_add(contrib);
        });
    };
    if (spec) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            if (from_out)
                return submit([](T o, T z) { return Act<A>::df_out(o, z); });
            return submit([](T, T z) { return Act<A>::df(z); });
        });
    }
    if (from_out)
        return submit([=](T o, T z) { return derivative_from_output(act, o, z); });
    return submit([=](T, T z) { return derivative(act, z); });
}

/// delta_l = (W_{l+1}^T delta_{l+1}) * f'(net_l)   (storage index l = layer l+1)
template <typename T> sycl::event Network<T>::hidden_delta(std::size_t l, std::size_t B, const ev_list &deps) {
    const std::size_t M = m_layers[l + 1].neurons; // neurons of this hidden layer
    const std::size_t K = m_layers[l + 2].neurons; // neurons of the next layer
    T *delta = m_delta[l].data();
    sycl::event ge = m_blas.gemm(m_queue, Blas::T_, Blas::N, std::int64_t(M), std::int64_t(B), std::int64_t(K), T(1),
                                 m_W[l + 1].data(), std::int64_t(K), m_delta[l + 1].data(), std::int64_t(K), T(0), delta,
                                 std::int64_t(M), deps);
    m_prof.record(Phase::Gemm, ge);
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[l + 1].data();
    const ActivationType act = m_layers[l + 1].activation;
    const bool from_out = m_opts.derivative_from_output;
    if (m_opts.specialized_kernels) {
        return dispatch_activation(act, [&](auto tag) {
            constexpr ActivationType A = decltype(tag)::value;
            if (from_out)
                return launch(n, {ge}, Phase::Delta, [=](std::size_t i) { delta[i] *= Act<A>::df_out(out[i], net[i]); });
            return launch(n, {ge}, Phase::Delta, [=](std::size_t i) { delta[i] *= Act<A>::df(net[i]); });
        });
    }
    if (from_out)
        return launch(n, {ge}, Phase::Delta, [=](std::size_t i) { delta[i] *= derivative_from_output(act, out[i], net[i]); });
    return launch(n, {ge}, Phase::Delta, [=](std::size_t i) { delta[i] *= derivative(act, net[i]); });
}

/// gW_l = delta_l in^T / B ; gb_l = sum_batch delta_l / B
template <typename T>
void Network<T>::gradients(std::size_t l, const T *in, std::size_t B, const ev_list &deps, sycl::event &ev_w, sycl::event &ev_b) {
    const std::size_t M = m_layers[l + 1].neurons, N = m_layers[l].neurons;
    const T *delta = m_delta[l].data();
    const T invB = T(1) / static_cast<T>(B);
    ev_w = m_blas.gemm(m_queue, Blas::N, Blas::T_, std::int64_t(M), std::int64_t(N), std::int64_t(B), invB, delta,
                       std::int64_t(M), in, std::int64_t(N), T(0), m_gW[l].data(), std::int64_t(M), deps);
    m_prof.record(Phase::Gemm, ev_w);
    T *gb = m_gb[l].data();
    if (m_opts.bias_gemv) {
        ev_b = m_blas.gemv(m_queue, Blas::N, std::int64_t(M), std::int64_t(B), invB, delta, std::int64_t(M), m_ones.data(),
                           1, T(0), gb, 1, deps);
        m_prof.record(Phase::BiasGrad, ev_b);
    } else {
        ev_b = launch(M, deps, Phase::BiasGrad, [=](std::size_t j) {
            T s = T(0);
            for (std::size_t n = 0; n < B; ++n)
                s += delta[j + M * n];
            gb[j] = s / static_cast<T>(B);
        });
    }
}

/// One fused launch over weights and biases of layer l.
template <typename T>
sycl::event Network<T>::update(std::size_t l, unsigned adam_step, unsigned epoch, unsigned max_epochs, const ev_list &deps) {
    using Strategy = typename AdaptiveLearningRate<T>::Strategy;
    const Strategy strategy = m_adapt.strategy;
    const bool momentum = m_mom.type == MomentumConfig<T>::Type::Classical;
    const T mu = m_mom.momentum_rate;
    const T lr = m_lr;
    const T eps = m_adapt.epsilon, beta1 = m_adapt.beta1, beta2 = m_adapt.beta2;
    const bool l1 = m_reg.uses_l1(), l2 = m_reg.uses_l2();
    const T lambda1 = m_reg.lambda1, lambda2 = m_reg.lambda2;
    T lr_eff = lr;
    if (strategy == Strategy::LinearDecay) {
        const T gamma = (max_epochs > 1) ? static_cast<T>(epoch) / static_cast<T>(max_epochs - 1) : T(0);
        lr_eff = lr * (T(1) - gamma) + gamma * m_adapt.final_lr;
    }
    // Adam bias-correction factors, once per step on the host (0.1: sycl::pow per weight)
    const bool host_corr = m_opts.host_adam_correction;
    const bool adam = strategy == Strategy::Adam;
    const T c1 = (host_corr && adam) ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(beta1), adam_step))) : T(0);
    const T c2 = (host_corr && adam) ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(beta2), adam_step))) : T(0);
    const T step_t = static_cast<T>(adam_step);

    const std::size_t nw = m_W[l].size(), nb = m_b[l].size();
    T *W = m_W[l].data(), *b = m_b[l].data();
    const T *gW = m_gW[l].data(), *gb = m_gb[l].data();
    T *mW = m_mW[l].data(), *vW = m_vW[l].data(), *mb = m_mb[l].data(), *vb = m_vb[l].data();

    auto step_fn = [=](T &p, T grad, T reg_grad, T &m, T &v) {
        const T g = grad - reg_grad; // ascent direction of (t - o)
        T step;
        switch (strategy) {
        case Strategy::Constant:
        case Strategy::LinearDecay:
            if (momentum) {
                m = mu * m + lr_eff * g;
                step = m;
            } else {
                step = lr_eff * g;
            }
            break;
        case Strategy::AdaGrad: {
            const T G = -g;
            v += G * G;
            step = (lr / (sycl::sqrt(v) + eps)) * g;
        } break;
        case Strategy::RMSProp: {
            const T G = -g;
            v = beta1 * v + (T(1) - beta1) * G * G;
            step = (lr / (sycl::sqrt(v) + eps)) * g;
        } break;
        case Strategy::Adam:
        default: {
            const T G = -g;
            m = beta1 * m + (T(1) - beta1) * G;
            v = beta2 * v + (T(1) - beta2) * G * G;
            T m_hat, v_hat;
            if (host_corr) {
                m_hat = m * c1;
                v_hat = v * c2;
            } else {
                m_hat = m / (T(1) - sycl::pow(beta1, step_t));
                v_hat = v / (T(1) - sycl::pow(beta2, step_t));
            }
            step = -(lr * m_hat / (sycl::sqrt(v_hat) + eps));
        } break;
        }
        p += step;
    };
    return launch(nw + nb, deps, Phase::Update, [=](std::size_t i) {
        if (i < nw) {
            const T w = W[i];
            T r = T(0);
            if (l1)
                r += lambda1 * sgn(w);
            if (l2)
                r += lambda2 * w;
            step_fn(W[i], gW[i], r, mW[i], vW[i]);
        } else {
            const std::size_t j = i - nw;
            step_fn(b[j], gb[j], T(0), mb[j], vb[j]);
        }
    });
}

/// penalty = sum_l lambda1 |W_l|_1 + 0.5 lambda2 |W_l|_2^2  -> m_loss_acc_d[1]
template <typename T> sycl::event Network<T>::penalty(const ev_list &deps) {
    const bool l1 = m_reg.uses_l1(), l2 = m_reg.uses_l2();
    ev_list evs;
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::int64_t n = std::int64_t(m_W[l].size());
        if (l1) {
            auto e = m_blas.asum(m_queue, n, m_W[l].data(), 1, m_reg_tmp.data() + 2 * l, {deps[l]});
            m_prof.record(Phase::Reg, e);
            evs.push_back(e);
        }
        if (l2) {
            auto e = m_blas.nrm2(m_queue, n, m_W[l].data(), 1, m_reg_tmp.data() + 2 * l + 1, {deps[l]});
            m_prof.record(Phase::Reg, e);
            evs.push_back(e);
        }
    }
    const T lambda1 = m_reg.lambda1, lambda2 = m_reg.lambda2;
    const std::size_t L = m_L;
    const T *tmp = m_reg_tmp.data();
    double *acc = m_loss_acc_d.data();
    auto e = m_queue.submit([&](sycl::handler &h) {
        h.depends_on(evs);
        h.single_task([=] {
            double p = 0.0;
            for (std::size_t l = 0; l < L; ++l) {
                if (l1)
                    p += static_cast<double>(lambda1) * static_cast<double>(tmp[2 * l]);
                if (l2) {
                    const double nrm = static_cast<double>(tmp[2 * l + 1]);
                    p += 0.5 * static_cast<double>(lambda2) * nrm * nrm;
                }
            }
            acc[1] = p;
        });
    });
    m_prof.record(Phase::Reg, e);
    return e;
}

// ---------------------------------------------------------------- training

template <typename T>
std::vector<T> Network<T>::train(const std::vector<T> &input_samples, const std::vector<T> &target_samples,
                                 unsigned num_samples, unsigned batch_size, unsigned max_epochs) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        throw std::invalid_argument("syclnn: number of samples cannot be zero");
    if (batch_size == 0)
        throw std::invalid_argument("syclnn: batch size must be greater than 0");
    if (max_epochs == 0)
        throw std::invalid_argument("syclnn: max_epochs must be greater than 0");
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("syclnn: input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in) + " (num_samples x " +
                                    std::to_string(n_in) + ")");
    if (target_samples.size() != std::size_t(num_samples) * n_out)
        throw std::invalid_argument("syclnn: targets have " + std::to_string(target_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_out) + " (num_samples x " +
                                    std::to_string(n_out) + ")");
    const std::size_t N = num_samples;
    const std::size_t B = std::min<std::size_t>(batch_size, N);
    const std::size_t n_batches = (N + B - 1) / B;
    const bool fine = m_opts.fine_deps;
    const bool direct = m_opts.direct_input;
    const bool use_reg = m_reg.uses_l1() || m_reg.uses_l2();

    ensure_workspace(B);
    m_adam_step = 0;

    // ---- dataset upload (once) ----
    // Buffers are declared *before* the guard so that, if anything throws, the
    // guard's destructor drains the queue before the buffers are released.
    Buf X, Y, Xs, Ys, stage_x, stage_y;
    detail::UsmBuffer<std::uint32_t> perm_dev;
    std::vector<std::uint32_t> perm_host;
    struct Quiesce {
        sycl::queue &q;
        ~Quiesce() {
            if (std::uncaught_exceptions() > 0) {
                try {
                    q.wait();
                } catch (...) {
                }
            }
        }
    } quiesce{m_queue};
    ev_list ev_ds;
    {
        X = Buf(N * n_in, m_queue, m_kind);
        Y = Buf(N * n_out, m_queue, m_kind);
        if (m_opts.shuffle) {
            Xs = Buf(N * n_in, m_queue, m_kind);
            Ys = Buf(N * n_out, m_queue, m_kind);
            perm_dev = detail::UsmBuffer<std::uint32_t>(N, m_queue, m_kind);
        }
        const T *src_x = input_samples.data();
        const T *src_y = target_samples.data();
        if (m_opts.pinned_host) {
            const auto t0 = Profiler::clock::now();
            stage_x = Buf(N * n_in, m_queue, MemoryKind::Host);
            stage_y = Buf(N * n_out, m_queue, MemoryKind::Host);
            std::memcpy(stage_x.data(), src_x, N * n_in * sizeof(T));
            std::memcpy(stage_y.data(), src_y, N * n_out * sizeof(T));
            m_prof.profile().h2d_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
            src_x = stage_x.data();
            src_y = stage_y.data();
        }
        h2d(X.data(), src_x, N * n_in, {}, &ev_ds);
        h2d(Y.data(), src_y, N * n_out, {}, &ev_ds);
    }
    std::mt19937 shuffle_rng(m_opts.shuffle_seed);

    std::vector<T> losses;
    losses.reserve(max_epochs);
    ev_list ev_upd(m_L), ev_act(m_L + 1), ev_delta(m_L), ev_gw(m_L), ev_gb(m_L);
    ev_list chain_next; // coarse mode: the updates of the previous batch
    ev_list ev_fill;

    for (unsigned epoch = 0; epoch < max_epochs; ++epoch) {
        const auto t_epoch = Profiler::clock::now();
        // reset the accumulators (the queue is idle here: every epoch ends with a wait)
        ev_fill.clear();
        ev_fill.push_back(m_queue.fill(m_loss_acc_d.data(), 0.0, 2));
        m_prof.record(Phase::Other, ev_fill.back());
        if (!m_opts.loss_reduction) {
            ev_fill.push_back(m_queue.fill(m_loss_acc_t.data(), T(0), 1));
            m_prof.record(Phase::Other, ev_fill.back());
        }
        const T *Xsrc = X.data();
        const T *Ysrc = Y.data();
        ev_list ev_data = ev_ds;
        if (m_opts.shuffle) {
            perm_host = detail::shuffle_permutation(static_cast<std::uint32_t>(N), shuffle_rng);
            auto ep = m_queue.memcpy(perm_dev.data(), perm_host.data(), N * sizeof(std::uint32_t), ev_ds);
            m_prof.record(Phase::H2D, ep, N * sizeof(std::uint32_t));
            const std::uint32_t *perm = perm_dev.data();
            const T *x0 = X.data();
            const T *y0 = Y.data();
            T *xs = Xs.data();
            T *ys = Ys.data();
            ev_list gather_deps = ev_ds;
            gather_deps.push_back(ep);
            // previous epoch fully done (wait at epoch end), so no WAR hazard on Xs/Ys
            auto gx = launch(N * n_in, gather_deps, Phase::Other, [=](std::size_t i) {
                const std::size_t c = i / n_in, r = i - c * n_in;
                xs[i] = x0[r + n_in * perm[c]];
            });
            auto gy = launch(N * n_out, gather_deps, Phase::Other, [=](std::size_t i) {
                const std::size_t c = i / n_out, r = i - c * n_out;
                ys[i] = y0[r + n_out * perm[c]];
            });
            ev_data = {gx, gy};
            Xsrc = Xs.data();
            Ysrc = Ys.data();
        }
        ev_list first_deps = ev_data;
        first_deps.insert(first_deps.end(), ev_fill.begin(), ev_fill.end());

        for (std::size_t bi = 0; bi < n_batches; ++bi) {
            const std::size_t start = bi * B;
            const std::size_t cur = std::min(B, N - start);
            if (m_adapt.strategy == AdaptiveLearningRate<T>::Strategy::Adam)
                ++m_adam_step;
            const T *x_batch = Xsrc + start * n_in;
            const T *targets = Ysrc + start * n_out;

            if (fine) {
                // ---- forward ----
                const T *in = x_batch;
                ev_list deps0 = (bi == 0) ? first_deps : ev_list{};
                if (!direct) {
                    ev_list cdeps = deps0;
                    cdeps.push_back(ev_upd[0]);
                    auto ce = m_queue.memcpy(m_act[0].data(), x_batch, cur * n_in * sizeof(T), cdeps);
                    m_prof.record(Phase::Other, ce);
                    deps0 = {ce};
                    in = m_act[0].data();
                }
                for (std::size_t l = 0; l < m_L; ++l) {
                    ev_list deps = (l == 0) ? deps0 : ev_list{ev_act[l]};
                    deps.push_back(ev_upd[l]);
                    if (l + 1 < m_L)
                        deps.push_back(ev_upd[l + 1]);
                    ev_act[l + 1] = forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, deps, nullptr);
                }
                // ---- output delta + loss ----
                {
                    ev_list deps{ev_act[m_L]};
                    if (bi == 0)
                        deps.insert(deps.end(), ev_fill.begin(), ev_fill.end());
                    ev_delta[m_L - 1] = output_delta_loss(targets, cur, deps);
                }
                // ---- hidden deltas ----
                for (std::size_t l = m_L - 1; l-- > 0;)
                    ev_delta[l] = hidden_delta(l, cur, {ev_delta[l + 1]});
                // ---- gradients and updates ----
                for (std::size_t l = 0; l < m_L; ++l) {
                    gradients(l, (l == 0) ? in : m_act[l].data(), cur, {ev_delta[l]}, ev_gw[l], ev_gb[l]);
                    ev_list deps{ev_gw[l], ev_gb[l]};
                    if (l >= 1)
                        deps.push_back(ev_delta[l - 1]); // the hidden-delta GEMM of layer l reads W_l
                    ev_upd[l] = update(l, m_adam_step, epoch, max_epochs, deps);
                }
            } else {
                // ---- 0.1-style coarse graph: one chain through the phases ----
                ev_list chain = (bi == 0) ? first_deps : chain_next;
                const T *in = x_batch;
                if (!direct) {
                    auto ce = m_queue.memcpy(m_act[0].data(), x_batch, cur * n_in * sizeof(T), chain);
                    m_prof.record(Phase::Other, ce);
                    chain = {ce};
                    in = m_act[0].data();
                }
                for (std::size_t l = 0; l < m_L; ++l)
                    chain = {forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, chain, nullptr)};
                chain = {output_delta_loss(targets, cur, chain)};
                ev_delta[m_L - 1] = chain.front();
                for (std::size_t l = m_L - 1; l-- > 0;) {
                    chain = {hidden_delta(l, cur, chain)};
                    ev_delta[l] = chain.front();
                }
                ev_list grads;
                for (std::size_t l = 0; l < m_L; ++l) {
                    gradients(l, (l == 0) ? in : m_act[l].data(), cur, chain, ev_gw[l], ev_gb[l]);
                    grads.push_back(ev_gw[l]);
                    grads.push_back(ev_gb[l]);
                }
                const ev_list all_grads = join(grads); // 0.1: empty join kernel; 0.2: multi-event dependency
                ev_list updates;
                for (std::size_t l = 0; l < m_L; ++l) {
                    ev_upd[l] = update(l, m_adam_step, epoch, max_epochs, all_grads);
                    updates.push_back(ev_upd[l]);
                }
                chain_next = join(updates);
            }
        }

        // ---- epoch end: penalty, loss readback, synchronisation ----
        ev_list tail;
        if (use_reg) {
            tail.push_back(penalty(ev_upd));
        }
        ev_list rb_deps = tail;
        rb_deps.insert(rb_deps.end(), ev_upd.begin(), ev_upd.end());
        rb_deps.push_back(ev_delta[m_L - 1]);
        sycl::event rb = m_queue.memcpy(m_host_scalars.data(), m_loss_acc_d.data(), 2 * sizeof(double), rb_deps);
        m_prof.record(Phase::D2H, rb, 2 * sizeof(double));
        sycl::event rb_t;
        if (!m_opts.loss_reduction) {
            rb_t = m_queue.memcpy(m_host_scalar_t.data(), m_loss_acc_t.data(), sizeof(T), rb_deps);
            m_prof.record(Phase::D2H, rb_t, sizeof(T));
        }
        m_prof.timed_wait([&] { m_queue.wait_and_throw(); });
        m_prof.flush();
        const double data_loss = m_opts.loss_reduction ? m_host_scalars.data()[0] : static_cast<double>(m_host_scalar_t.data()[0]);
        const double total = data_loss / static_cast<double>(N) + (use_reg ? m_host_scalars.data()[1] : 0.0);
        losses.push_back(static_cast<T>(total));
        if (m_opts.record_history)
            snapshot_history();
        if (m_prof.enabled()) {
            m_prof.profile().epoch_wall_ns.push_back(Profiler::elapsed_ns(t_epoch));
            ++m_prof.profile().epochs;
            m_prof.profile().batches += n_batches;
        }
        // ---- stop criteria ----
        bool stop = false;
        if (m_stop.type == StopCriteria<T>::Type::MinError) {
            stop = losses.back() < m_stop.threshold;
        } else if (m_stop.type == StopCriteria<T>::Type::MinErrorChange && epoch > 0) {
            stop = std::abs(losses[epoch - 1] - losses[epoch]) < m_stop.threshold;
        }
        if (stop)
            break;
    }
    wait_all();
    if (!m_opts.record_history)
        snapshot_history();
    if (!m_opts.persistent_workspace)
        release_workspace();
    if (m_prof.enabled())
        m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    return losses;
}

// ---------------------------------------------------------------- inference

template <typename T>
std::vector<T> Network<T>::predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        return {};
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("syclnn: predict input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in));
    const std::size_t N = num_samples;
    const std::size_t B = (batch_size == 0) ? N : std::min<std::size_t>(batch_size, N);
    ensure_workspace(B);

    std::vector<T> out(N * n_out);
    Buf X, stage_x, stage_out;
    struct Quiesce {
        sycl::queue &q;
        ~Quiesce() {
            if (std::uncaught_exceptions() > 0) {
                try {
                    q.wait();
                } catch (...) {
                }
            }
        }
    } quiesce{m_queue};
    ev_list ev_ds;
    const T *src = input_samples.data();
    X = Buf(N * n_in, m_queue, m_kind);
    if (m_opts.pinned_host) {
        const auto t0 = Profiler::clock::now();
        stage_x = Buf(N * n_in, m_queue, MemoryKind::Host);
        std::memcpy(stage_x.data(), src, N * n_in * sizeof(T));
        m_prof.profile().h2d_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
        src = stage_x.data();
        stage_out = Buf(N * n_out, m_queue, MemoryKind::Host);
    }
    h2d(X.data(), src, N * n_in, {}, &ev_ds);
    T *out_ptr = m_opts.pinned_host ? stage_out.data() : out.data();

    ev_list chain = ev_ds;
    sycl::event last_copy;
    for (std::size_t start = 0; start < N; start += B) {
        const std::size_t cur = std::min(B, N - start);
        const T *in = X.data() + start * n_in;
        if (!m_opts.direct_input) {
            auto ce = m_queue.memcpy(m_act[0].data(), in, cur * n_in * sizeof(T), chain);
            m_prof.record(Phase::Other, ce);
            chain = {ce};
            in = m_act[0].data();
        }
        for (std::size_t l = 0; l < m_L; ++l)
            chain = {forward_layer(l, (l == 0) ? in : m_act[l].data(), cur, chain, nullptr)};
        last_copy = m_queue.memcpy(out_ptr + start * n_out, m_act[m_L].data(), cur * n_out * sizeof(T), chain);
        m_prof.record(Phase::D2H, last_copy, cur * n_out * sizeof(T));
        chain = {last_copy}; // the workspace is reused by the next chunk
    }
    m_prof.timed_wait([&] { m_queue.wait_and_throw(); });
    m_prof.flush();
    if (m_opts.pinned_host) {
        const auto t0 = Profiler::clock::now();
        std::memcpy(out.data(), stage_out.data(), N * n_out * sizeof(T));
        m_prof.profile().d2h_ns += m_prof.enabled() ? Profiler::elapsed_ns(t0) : 0;
    }
    if (!m_opts.persistent_workspace)
        release_workspace();
    if (m_prof.enabled())
        m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    return out;
}

} // namespace syclnn
