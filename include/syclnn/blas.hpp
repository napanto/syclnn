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

enum class BlasBackend { Auto, MklCpu, Netlib, Generic, Cublas, Rocblas };

inline const char *blas_backend_name(BlasBackend b) {
    switch (b) {
    case BlasBackend::Auto: return "auto";
    case BlasBackend::MklCpu: return "mklcpu";
    case BlasBackend::Netlib: return "netlib";
    case BlasBackend::Generic: return "generic";
    case BlasBackend::Cublas: return "cublas";
    case BlasBackend::Rocblas: return "rocblas";
    }
    return "?";
}

/// Backends this build was linked against (from oneMath's config.hpp macros).
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
    throw std::invalid_argument("unknown BLAS backend '" + name + "'");
}

namespace detail {
[[noreturn]] inline void not_compiled(BlasBackend b) {
    std::string avail;
    for (const auto &n : compiled_blas_backends())
        avail += (avail.empty() ? "" : ", ") + n;
    throw std::invalid_argument(std::string("oneMath backend '") + blas_backend_name(b) +
                                "' is not compiled into this build (available: " + avail + ")");
}

/// Invoke `fn(selector)` where selector is the queue (run-time dispatch) or a
/// compile-time backend_selector.
template <typename Fn> inline sycl::event with_backend(BlasBackend b, sycl::queue &q, Fn &&fn) {
    namespace om = oneapi::math;
    switch (b) {
    case BlasBackend::Auto: return fn(q);
    case BlasBackend::MklCpu:
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
        return fn(om::backend_selector<om::backend::mklcpu>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Netlib:
#ifdef ONEMATH_ENABLE_NETLIB_BACKEND
        return fn(om::backend_selector<om::backend::netlib>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Generic:
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND
        return fn(om::backend_selector<om::backend::generic>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Cublas:
#ifdef ONEMATH_ENABLE_CUBLAS_BACKEND
        return fn(om::backend_selector<om::backend::cublas>{q});
#else
        not_compiled(b);
#endif
    case BlasBackend::Rocblas:
#ifdef ONEMATH_ENABLE_ROCBLAS_BACKEND
        return fn(om::backend_selector<om::backend::rocblas>{q});
#else
        not_compiled(b);
#endif
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
        case BlasBackend::Generic: return;
        }
    }

    template <typename T>
    sycl::event gemm(sycl::queue &q, transpose ta, transpose tb, std::int64_t m, std::int64_t n, std::int64_t k, T alpha,
                     const T *a, std::int64_t lda, const T *b, std::int64_t ldb, T beta, T *c, std::int64_t ldc,
                     const std::vector<sycl::event> &deps = {}) const {
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::gemm(sel, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, deps);
        });
    }

    template <typename T>
    sycl::event gemv(sycl::queue &q, transpose ta, std::int64_t m, std::int64_t n, T alpha, const T *a, std::int64_t lda,
                     const T *x, std::int64_t incx, T beta, T *y, std::int64_t incy,
                     const std::vector<sycl::event> &deps = {}) const {
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::gemv(sel, ta, m, n, alpha, a, lda, x, incx, beta, y, incy, deps);
        });
    }

    template <typename T>
    sycl::event asum(sycl::queue &q, std::int64_t n, const T *x, std::int64_t incx, T *result,
                     const std::vector<sycl::event> &deps = {}) const {
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::asum(sel, n, x, incx, result, deps);
        });
    }

    template <typename T>
    sycl::event nrm2(sycl::queue &q, std::int64_t n, const T *x, std::int64_t incx, T *result,
                     const std::vector<sycl::event> &deps = {}) const {
        return detail::with_backend(m_backend, q, [&](auto sel) {
            return oneapi::math::blas::column_major::nrm2(sel, n, x, incx, result, deps);
        });
    }

  private:
    BlasBackend m_backend;
};

} // namespace syclnn
