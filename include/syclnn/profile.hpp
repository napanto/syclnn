// SPDX-License-Identifier: LGPL-3.0-only
// syclnn - per-phase profiler fed by SYCL event profiling info.
#pragma once

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "syclnn/config.hpp"

namespace syclnn {

enum class Phase { H2D, D2H, Gemm, Act, Delta, BiasGrad, Update, Loss, Reg, Other };

/**
 * Collects the events submitted by Network and, at synchronisation points,
 * folds their command_start/command_end timestamps into a Profile.  When
 * disabled it only counts launches and bytes (no event is retained), so the
 * hot path stays free of profiling calls.
 */
class Profiler {
  public:
    explicit Profiler(bool enabled = false) : m_enabled(enabled) {}

    bool enabled() const { return m_enabled; }
    Profile &profile() { return m_profile; }
    const Profile &profile() const { return m_profile; }
    void reset() {
        m_profile.reset();
        m_pending.clear();
    }

    /// Register a submitted operation.  `bytes` counts host<->device traffic.
    void record(Phase phase, const sycl::event &ev, std::uint64_t bytes = 0) {
        if (!m_enabled)
            return;
        ++m_profile.launches;
        if (phase == Phase::H2D)
            m_profile.bytes_h2d += bytes;
        else if (phase == Phase::D2H)
            m_profile.bytes_d2h += bytes;
        m_pending.emplace_back(phase, ev);
    }

    /// Fold every pending event into the profile (call after a wait()).
    void flush() {
        if (!m_enabled)
            return;
        for (auto &[phase, ev] : m_pending) {
            std::uint64_t ns = 0;
            try {
                const auto start = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto end = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
                ns = end >= start ? end - start : 0;
            } catch (const sycl::exception &) {
                ++m_profile.unprofiled;
                continue;
            }
            slot(phase) += ns;
        }
        m_pending.clear();
    }

    /// Time a blocking call (wait / wait_and_throw).
    template <typename F> void timed_wait(F &&f) {
        const auto t0 = clock::now();
        f();
        m_profile.wait_ns += elapsed_ns(t0);
    }

    using clock = std::chrono::steady_clock;
    static std::uint64_t elapsed_ns(clock::time_point since) {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - since).count());
    }

  private:
    std::uint64_t &slot(Phase p) {
        switch (p) {
        case Phase::H2D: return m_profile.h2d_ns;
        case Phase::D2H: return m_profile.d2h_ns;
        case Phase::Gemm: return m_profile.gemm_ns;
        case Phase::Act: return m_profile.act_ns;
        case Phase::Delta: return m_profile.delta_ns;
        case Phase::BiasGrad: return m_profile.biasgrad_ns;
        case Phase::Update: return m_profile.update_ns;
        case Phase::Loss: return m_profile.loss_ns;
        case Phase::Reg: return m_profile.reg_ns;
        case Phase::Other:
        default: return m_profile.other_ns;
        }
    }

    bool m_enabled;
    Profile m_profile;
    std::vector<std::pair<Phase, sycl::event>> m_pending;
};

} // namespace syclnn
