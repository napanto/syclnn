// SPDX-License-Identifier: LGPL-3.0-only
// syclnn - device enumeration and selection.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "syclnn/config.hpp"

namespace syclnn {

/// Description of one SYCL device as seen by this build of the library.
struct DeviceInfo {
    int index = 0;            ///< position in devices() / "index:N"
    std::string name;         ///< info::device::name
    std::string vendor;       ///< info::device::vendor
    std::string type;         ///< "cpu", "gpu", "accelerator", "host"
    std::string backend;      ///< "opencl", "level_zero", "cuda", "hip", "omp", ...
    std::string platform;     ///< info::platform::name
    std::string driver;       ///< info::device::driver_version
    std::uint64_t global_mem_bytes = 0;
    unsigned compute_units = 0;
    bool fp64 = false;
};

inline std::string device_type_name(sycl::info::device_type t) {
    switch (t) {
    case sycl::info::device_type::cpu: return "cpu";
    case sycl::info::device_type::gpu: return "gpu";
    case sycl::info::device_type::accelerator: return "accelerator";
    case sycl::info::device_type::host: return "host";
    default: return "other";
    }
}

inline std::string backend_name(const sycl::device &d) {
    const sycl::backend b = d.get_backend();
#if defined(__ADAPTIVECPP__) || defined(__HIPSYCL__)
    switch (b) {
    case sycl::backend::omp: return "omp";
    case sycl::backend::cuda: return "cuda";
    case sycl::backend::hip: return "hip";
    case sycl::backend::ocl: return "opencl";
    case sycl::backend::level_zero: return "level_zero";
    default: return "unknown";
    }
#else
    switch (b) {
    case sycl::backend::opencl: return "opencl";
    case sycl::backend::ext_oneapi_level_zero: return "level_zero";
    case sycl::backend::ext_oneapi_cuda: return "cuda";
    case sycl::backend::ext_oneapi_hip: return "hip";
#ifdef SYCL_EXT_ONEAPI_NATIVE_CPU
    case sycl::backend::ext_oneapi_native_cpu: return "native_cpu";
#endif
    default: return "unknown";
    }
#endif
}

inline std::vector<sycl::device> all_devices() { return sycl::device::get_devices(); }

inline DeviceInfo describe(const sycl::device &d, int index) {
    DeviceInfo info;
    info.index = index;
    info.name = d.get_info<sycl::info::device::name>();
    info.vendor = d.get_info<sycl::info::device::vendor>();
    info.type = device_type_name(d.get_info<sycl::info::device::device_type>());
    info.backend = backend_name(d);
    info.platform = d.get_platform().get_info<sycl::info::platform::name>();
    info.driver = d.get_info<sycl::info::device::driver_version>();
    info.global_mem_bytes = d.get_info<sycl::info::device::global_mem_size>();
    info.compute_units = d.get_info<sycl::info::device::max_compute_units>();
    info.fp64 = d.has(sycl::aspect::fp64);
    return info;
}

/// All devices visible to the SYCL runtime, in "index:N" order.
inline std::vector<DeviceInfo> devices() {
    std::vector<DeviceInfo> out;
    const auto devs = all_devices();
    out.reserve(devs.size());
    for (std::size_t i = 0; i < devs.size(); ++i)
        out.push_back(describe(devs[i], static_cast<int>(i)));
    return out;
}

namespace detail {
inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}
inline bool is_number(const std::string &s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}
} // namespace detail

/**
 * Resolve an Options::device string:
 *   "default"                       -> sycl::default_selector_v
 *   "cpu" | "gpu" | "accelerator"    -> first device of that type
 *   "index:N" | "N"                 -> N-th device of devices()
 *   "<backend>:N"                   -> N-th device of that backend (cuda:0, opencl:1, hip:0, ...)
 *   "<backend>:<type>"              -> first device of that type on that backend (opencl:cpu)
 *   anything else                   -> case-insensitive substring of the device name
 */
inline sycl::device select_device(const std::string &spec) {
    const std::string s = detail::lower(spec);
    if (s.empty() || s == "default" || s == "auto")
        return sycl::device(sycl::default_selector_v);
    const auto devs = all_devices();
    auto no_device = [&](const std::string &what) {
        std::string names;
        for (std::size_t i = 0; i < devs.size(); ++i)
            names += "\n  index:" + std::to_string(i) + "  " + backend_name(devs[i]) + ":" +
                     device_type_name(devs[i].get_info<sycl::info::device::device_type>()) + "  " +
                     devs[i].get_info<sycl::info::device::name>();
        return std::invalid_argument("no SYCL device matches '" + spec + "' (" + what + "); available:" + names);
    };
    if (s == "cpu" || s == "gpu" || s == "accelerator" || s == "host") {
        for (const auto &d : devs)
            if (device_type_name(d.get_info<sycl::info::device::device_type>()) == s)
                return d;
        throw no_device("device type");
    }
    std::string key = s, arg;
    const auto colon = s.find(':');
    if (colon != std::string::npos) {
        key = s.substr(0, colon);
        arg = s.substr(colon + 1);
    }
    if (detail::is_number(s) || (key == "index" && detail::is_number(arg))) {
        const std::size_t n = std::stoul(detail::is_number(s) ? s : arg);
        if (n >= devs.size())
            throw no_device("index out of range");
        return devs[n];
    }
    if (colon != std::string::npos) {
        std::vector<sycl::device> same;
        for (const auto &d : devs)
            if (backend_name(d) == key)
                same.push_back(d);
        if (!same.empty()) {
            if (detail::is_number(arg)) {
                const std::size_t n = std::stoul(arg);
                if (n < same.size())
                    return same[n];
                throw no_device("backend index out of range");
            }
            for (const auto &d : same)
                if (device_type_name(d.get_info<sycl::info::device::device_type>()) == arg)
                    return d;
            throw no_device("no such device type on that backend");
        }
    }
    for (const auto &d : devs)
        if (detail::lower(d.get_info<sycl::info::device::name>()).find(s) != std::string::npos)
            return d;
    throw no_device("no name match");
}

} // namespace syclnn
