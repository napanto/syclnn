// SPDX-License-Identifier: LGPL-3.0-only
// syclnn - thin wrapper over oneMath BLAS with run-time selectable backend.
//
// oneMath offers two dispatch modes: the run-time loader (`gemm(queue, ...)`,
// backend chosen by device type: MKLCPU/NETLIB/generic for CPUs, cuBLAS for
// NVIDIA, rocBLAS for AMD) and the compile-time selector
// (`gemm(backend_selector<backend::X>{queue}, ...)`).  Options::blas picks one
// of them, so the same wheel can be benchmarked with every CPU BLAS backend it
// was linked against (E1 of the study).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <oneapi/math.hpp>
#include <oneapi/math/blas.hpp>

namespace syclnn {

enum class BlasBackend { Auto, MklCpu, Netlib, Generic, Cublas, Rocblas, Tiled };

inline const char *blas_backend_name(BlasBackend b) {
    switch (b) {
    case BlasBackend::Auto: return "auto";
    case BlasBackend::MklCpu: return "mklcpu";
    case BlasBackend::Netlib: return "netlib";
    case BlasBackend::Generic: return "generic";
    case BlasBackend::Cublas: return "cublas";
    case BlasBackend::Rocblas: return "rocblas";
    case BlasBackend::Tiled: return "tiled";
    }
    return "?";
}

/// Backends the oneMath build knows (run-time loader), from its config.hpp macros.
inline std::vector<std::string> compiled_blas_backends() {
    std::vector<std::string> v;
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
    v.push_back("mklcpu");
#endif
#ifdef ONEMATH_ENABLE_NETLIB_BACKEND
    v.push_back("netlib");
#endif
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND
    v.push_back("generic");
#endif
#ifdef ONEMATH_ENABLE_CUBLAS_BACKEND
    v.push_back("cublas");
#endif
#ifdef ONEMATH_ENABLE_ROCBLAS_BACKEND
    v.push_back("rocblas");
#endif
    return v;
}

/// Backends that Options::blas can name explicitly: those linked for oneMath's
/// compile-time dispatch (SYCLNN_CT_BACKENDS at build time). GPU backends are
/// normally reached through "auto" so that the module does not depend on the
/// vendor driver library on machines without that GPU.
inline std::vector<std::string> selectable_blas_backends() {
    std::vector<std::string> v;
#ifdef SYCLNN_CT_MKLCPU
    v.push_back("mklcpu");
#endif
#ifdef SYCLNN_CT_NETLIB
    v.push_back("netlib");
#endif
#ifdef SYCLNN_CT_GENERIC
    v.push_back("generic");
#endif
#ifdef SYCLNN_CT_CUBLAS
    v.push_back("cublas");
#endif
#ifdef SYCLNN_CT_ROCBLAS
    v.push_back("rocblas");
#endif
    v.push_back("tiled");
    return v;
}

inline BlasBackend parse_blas_backend(const std::string &name) {
    if (name.empty() || name == "auto")
        return BlasBackend::Auto;
    if (name == "mklcpu" || name == "mkl")
        return BlasBackend::MklCpu;
    if (name == "netlib" || name == "openblas")
        return BlasBackend::Netlib;
    if (name == "generic" || name == "portblas")
        return BlasBackend::Generic;
    if (name == "cublas")
        return BlasBackend::Cublas;
    if (name == "rocblas")
        return BlasBackend::Rocblas;
    if (name == "tiled" || name == "handwritten")
        return BlasBackend::Tiled;
    throw std::invalid_argument("unknown BLAS backend '" + name + "'");
}

/// Hand-written SYCL BLAS (Options::blas = "tiled"): a 16x16 local-memory tiled
/// GEMM (the classic shared-memory tiling of the lectures, identical in cudann's
/// CUDA and ompnn's OpenMP versions), a one-work-item-per-row GEMV and
/// reductions for asum/nrm2.  Column-major; work-item dimension 1 runs along the
/// rows so that global loads and stores are coalesced.
namespace handwritten {
constexpr int TILE = 16;

template <typename T>
inline sycl::event gemm(sycl::queue &q, bool ta, bool tb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha,
                        const T *A, std::int64_t lda, const T *B, std::int64_t ldb, T beta, T *C, std::int64_t ldc,
                        const std::vector<sycl::event> &deps) {
    const std::size_t gm = std::size_t((m + TILE - 1) / TILE) * TILE, gn = std::size_t((n + TILE - 1) / TILE) * TILE;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        // the op(A) tile is stored transposed (AsT[kk][li]): the fast work-item index li then
        // walks consecutive local-memory words in both the stores and the inner-loop loads
        // (As[li][kk] would be a stride-16 bank conflict); Bs[kk][lj] is a broadcast
        sycl::local_accessor<T, 2> AsT(sycl::range<2>(TILE, TILE), h), Bs(sycl::range<2>(TILE, TILE), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(gn, gm), sycl::range<2>(TILE, TILE)), [=](sycl::nd_item<2> it) {
            const int lj = int(it.get_local_id(0)), li = int(it.get_local_id(1));
            const std::int64_t j = std::int64_t(it.get_global_id(0)), i = std::int64_t(it.get_global_id(1));
            T acc = T(0);
            for (std::int64_t t = 0; t < k; t += TILE) {
                std::int64_t p = t + lj;
                AsT[lj][li] = (i < m && p < k) ? (ta ? A[p + i * lda] : A[i + p * lda]) : T(0);
                p = t + li;
                Bs[li][lj] = (p < k && j < n) ? (tb ? B[j + p * ldb] : B[p + j * ldb]) : T(0);
                sycl::group_barrier(it.get_group());
                for (int kk = 0; kk < TILE; ++kk)
                    acc += AsT[kk][li] * Bs[kk][lj];
                sycl::group_barrier(it.get_group());
            }
            if (i < m && j < n) {
                T *c = C + i + j * ldc;
                *c = (beta == T(0)) ? alpha * acc : alpha * acc + beta * (*c);
            }
        });
    });
}

template <typename T>
inline sycl::event gemv(sycl::queue &q, bool ta, std::int64_t m, std::int64_t n, T alpha, const T *A, std::int64_t lda,
                        const T *x, T beta, T *y, const std::vector<sycl::event> &deps) {
    const std::int64_t rows = ta ? n : m, inner = ta ? m : n;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(std::size_t(rows)), [=](sycl::item<1> it) {
            const std::int64_t i = std::int64_t(it.get_id(0));
            T acc = T(0);
            for (std::int64_t p = 0; p < inner; ++p)
                acc += (ta ? A[p + i * lda] : A[i + p * lda]) * x[p];
            y[i] = (beta == T(0)) ? alpha * acc : alpha * acc + beta * y[i];
        });
    });
}

/// result = sum |x| (abs) or sqrt(sum x^2) (!abs)
template <typename T>
inline sycl::event reduce(sycl::queue &q, bool abs, std::int64_t n, const T *x, T *result, const std::vector<sycl::event> &deps) {
    // initialize_to_identity: the reduction overwrites *result, no zero-fill kernel
    auto red = q.submit([&](sycl::handler &h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(std::size_t(n)),
                       sycl::reduction(result, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
                       [=](sycl::item<1> it, auto &sum) {
            const T v = x[it.get_id(0)];
            sum += abs ? (v < T(0) ? -v : v) : v * v;
        });
    });
    if (abs)
        return red;
    return q.submit([&](sycl::handler &h) {
        h.depends_on(red);
        h.single_task([=] { *result = sycl::sqrt(*result); });
    });
}
} // namespace handwritten

namespace detail {
[[noreturn]] inline void not_compiled(BlasBackend b) {
    std::string avail;
    for (const auto &n : selectable_blas_backends())
        avail += (avail.empty() ? "" : ", ") + n;
    throw std::invalid_argument(std::string("oneMath backend '") + blas_backend_name(b) +
                                "' cannot be selected in this build (selectable: auto" + (avail.empty() ? "" : ", " + avail) +
                                "; GPU backends are reached with blas=auto)");
}

/// Invoke `fn(selector)` where selector is the queue (run-time dispatch) or a
/// compile-time backend_selector.
template <typename Fn> inline sycl::event with_backend(BlasBackend b, sycl::queue &q, Fn &&fn) {
    namespace om = oneapi::math;
    switch (b) {
    case BlasBackend::Auto: return fn(q);
    case BlasBackend::MklCpu:
#if defined(SYCLNN_CT_MKLCPU) && defined(ONEMATH_ENABLE_MKLCPU_BACKEND)
        return fn(om::backend_selector<om::backend::mklcpu>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Netlib:
#if defined(SYCLNN_CT_NETLIB) && defined(ONEMATH_ENABLE_NETLIB_BACKEND)
        return fn(om::backend_selector<om::backend::netlib>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Generic:
#if defined(SYCLNN_CT_GENERIC) && defined(ONEMATH_ENABLE_GENERIC_BLAS_BACKEND)
        return fn(om::backend_selector<om::backend::generic>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Cublas:
#if defined(SYCLNN_CT_CUBLAS) && defined(ONEMATH_ENABLE_CUBLAS_BACKEND)
        return fn(om::backend_selector<om::backend::cublas>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Rocblas:
#if defined(SYCLNN_CT_ROCBLAS) && defined(ONEMATH_ENABLE_ROCBLAS_BACKEND)
        return fn(om::backend_selector<om::backend::rocblas>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Tiled:
        throw std::logic_error("syclnn: the tiled BLAS is dispatched by Blas, not by oneMath");
    }
    not_compiled(b);
}
} // namespace detail

/// Column-major BLAS calls on USM pointers, all asynchronous (return the event).
class Blas {
  public:
    using transpose = oneapi::math::transpose;
    static constexpr transpose N = transpose::nontrans;
    static constexpr transpose T_ = transpose::trans;

    explicit Blas(BlasBackend b = BlasBackend::Auto) : m_backend(b) {}
    BlasBackend backend() const { return m_backend; }

    /// Validate that the backend can serve `dev` (throws std::invalid_argument).
    void check(const sycl::device &dev) const {
        const bool cpu = dev.is_cpu();
        const bool gpu = dev.is_gpu();
        switch (m_backend) {
        case BlasBackend::Auto: return;
        case BlasBackend::MklCpu:
        case BlasBackend::Netlib:
            if (!cpu)
                throw std::invalid_argument(std::string(blas_backend_name(m_backend)) + " is a CPU BLAS backend");
            return;
        case BlasBackend::Cublas:
        case BlasBackend::Rocblas:
            if (!gpu)
                throw std::invalid_argument(std::string(blas_backend_name(m_backend)) + " is a GPU BLAS backend");
            return;
        case BlasBackend::Generic:
        case BlasBackend::Tiled: return;
        }
    }

    template <typename T>
    sycl::event gemm(sycl::queue &q, transpose ta, transpose tb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha,
                     const T *a, std::int64_t lda, const T *b, std::int64_t ldb, T beta, T *c, std::int64_t ldc,
                     const std::vector<sycl::event> &deps = {}) const {
        if (m_backend == BlasBackend::Tiled)
            return handwritten::gemm(q, ta == T_, tb == T_, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps);
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::gemm(sel, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps);
        });
    }

    template <typename T>
    sycl::event gemv(sycl::queue &q, transpose ta, std::int64_t m, std::int64_t n, T alpha, const T *a, std::int64_t lda,
                     const T *x, std::int64_t incx, T beta, T *y, std::int64_t incy,
                     const std::vector<sycl::event> &deps = {}) const {
        if (m_backend == BlasBackend::Tiled) {
            if (incx != 1 || incy != 1)
                throw std::invalid_argument("syclnn: the tiled BLAS supports unit strides only");
            return handwritten::gemv(q, ta == T_, m, n, alpha, a, lda, x, beta, y, deps);
        }
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::gemv(sel, ta, m, n, alpha, a, lda, x, incx, beta, y, incy, deps);
        });
    }

    template <typename T>
    sycl::event asum(sycl::queue &q, std::int64_t n, const T *x, std::int64_t incx, T *result,
                     const std::vector<sycl::event> &deps = {}) const {
        if (m_backend == BlasBackend::Tiled) {
            if (incx != 1)
                throw std::invalid_argument("syclnn: the tiled BLAS supports unit strides only");
            return handwritten::reduce(q, true, n, x, result, deps);
        }
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::asum(sel, n, x, incx, result, deps);
        });
    }

    template <typename T>
    sycl::event nrm2(sycl::queue &q, std::int64_t n, const T *x, std::int64_t incx, T *result,
                     const std::vector<sycl::event> &deps = {}) const {
        if (m_backend == BlasBackend::Tiled) {
            if (incx != 1)
                throw std::invalid_argument("syclnn: the tiled BLAS supports unit strides only");
            return handwritten::reduce(q, false, n, x, result, deps);
        }
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::nrm2(sel, n, x, incx, result, deps);
        });
    }

  private:
    BlasBackend m_backend;
};

} // namespace syclnn
