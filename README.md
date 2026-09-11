# syclnn

SYCL-accelerated feed-forward neural network library: mini-batch back-propagation
with SGD / momentum / linear decay / AdaGrad / RMSProp / Adam and L1 / L2 /
ElasticNet regularisation, for `float` and `double`, on any SYCL device
(CPU, NVIDIA, AMD, Intel GPUs). GEMM/GEMV go to
[oneMath](https://github.com/uxlfoundation/oneMath) (oneMKL, OpenBLAS/NETLIB,
generic SYCL BLAS, cuBLAS, rocBLAS); everything else is a hand-written SYCL kernel.
Header-only C++17 with a pybind11 module.

Version 0.2 is the SYCL side of the *SYCL vs CUDA vs OpenMP* study of the
Accelerated Computing course (University of Pisa); its siblings
[cudann](https://github.com/napanto/cudann) and [ompnn](https://github.com/napanto/ompnn)
implement the same API and numerics, and [fnn-bench](https://github.com/napanto/fnn-bench)
holds the NumPy oracle, the shared parity test-suite and the benchmark harness.
See `CHANGELOG.md` for what changed since version 0.1.

## Quick start

```python
import numpy as np, syclnn

layers = [syclnn.LayerDescription(17, syclnn.ActivationType.Disabled),
          syclnn.LayerDescription(8, syclnn.ActivationType.Tanh),
          syclnn.LayerDescription(1, syclnn.ActivationType.Sigmoid)]
net = syclnn.Network(layers, 0.05, dtype="double", device="gpu", profile=True, seed=1,
                     adaptive_learning_rate=syclnn.AdaptiveLearningRate_double(syclnn.Adam))
losses = net.train(X.ravel(), Y.ravel(), n_samples=len(X), batch_size=40, max_epochs=500)
pred = net.predict(Xtest.ravel(), n_samples=len(Xtest)).reshape(len(Xtest), -1)
print(net.device_name, net.profile["gemm_ns"] / 1e6, "ms in GEMM")
print(syclnn.devices())      # every visible SYCL device
print(syclnn.build_info())   # compiler, targets, oneMath backends of this build
```

`syclnn.Network_double` / `syclnn.Network_float` and the `*_double` / `*_float`
configuration classes are the 0.1 API and still work unchanged; the last
argument `options=syclnn.Options()` is new.

Inputs are `(N, n_in)` row-major, targets `(N, n_out)`; weights are exchanged as
flattened column-major `(n_out, n_in)` matrices, one array per layer.

### Options

| option | default | meaning |
|---|---|---|
| `device` | `"default"` | `cpu`, `gpu`, `index:N`, `cuda:0`, `opencl:cpu`, `hip:0`, or a name substring |
| `blas` | `"auto"` | oneMath backend: `mklcpu`, `netlib`, `generic`, `cublas`, `rocblas` (must be compiled in) |
| `profile` | `False` | per-phase device timings in `net.profile` |
| `record_history` | `False` | snapshot weights after every epoch (`net.weights_biases`) |
| `shuffle`, `shuffle_seed` | `False`, `1` | reshuffle samples every epoch (`std::mt19937` Fisher-Yates) |
| `memory` | `device` | `device` (explicit copies), `shared` (USM shared), `host` (zero-copy) |
| `queue` | `out_of_order` | `in_order` maps to one in-order queue |
| `pinned_host` | `True` | stage host<->device copies through pinned memory |
| ablations | | `loss_reduction`, `bias_gemv`, `direct_input`, `fine_deps`, `join_kernels`, `specialized_kernels`, `derivative_from_output`, `host_adam_correction`, `workgroup_size`, `persistent_workspace` (see CHANGELOG) |

## Run the published image

`ghcr.io/napanto/syclnn` is the library installed in the `fnn-sycl` toolchain image (DPC++, the
Intel OpenCL CPU runtime and oneMath with the MKLCPU, Netlib and cuBLAS backends), built by CI
from the `Containerfile` on every push. One command, no build:

```sh
# CPU only (any x86-64 host with a container runtime):
podman run --rm -it ghcr.io/napanto/syclnn python -c "import syclnn; print(syclnn.devices())"
# with an NVIDIA GPU (driver >= 525 and the NVIDIA container toolkit's CDI spec on the host):
podman run --rm -it --device nvidia.com/gpu=all ghcr.io/napanto/syclnn
```

The image targets `spir64` (any CPU) and `nvidia_gpu_sm_61` + `sm_80`; the AdaptiveCpp and AMD
builds are made from the `fnn-rocm` / `fnn-acpp-cuda` toolchain images (see `fnn-bench/containers`)
and are not published as library images.

## Building

Requirements: a SYCL compiler (open-source [intel/llvm](https://github.com/intel/llvm)
`clang++`, Intel `icpx`, or [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)),
oneMath v0.9 (pre-built, or fetched and built by CMake), CMake >= 3.26, Python >= 3.10.
The `ghcr.io/napanto/fnn-sycl` image (see `fnn-bench/containers`) has everything:
DPC++ v7.1.0 with the CUDA adapter, the Intel OpenCL CPU runtime, oneMath with
MKLCPU + NETLIB(OpenBLAS) + cuBLAS in `/opt/onemath` and the generic SYCL BLAS
build in `/opt/onemath-generic`.

```sh
# DPC++: one wheel for CPU (spir64 JIT) and NVIDIA (Pascal + Ampere)
export SYCLNN_TARGETS="spir64;nvidia_gpu_sm_61;nvidia_gpu_sm_80" SYCLNN_ONEMATH_ROOT=/opt/onemath
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -v .

# AdaptiveCpp (e.g. AMD GPUs, oneMath built with rocBLAS + NETLIB)
export SYCLNN_SYCL_IMPL=adaptivecpp SYCLNN_ONEMATH_ROOT=$HOME/.local/opt/fnn-rocm/onemath
CMAKE_ARGS="-DAdaptiveCpp_DIR=$HOME/.local/opt/fnn-rocm/acpp/lib/cmake/AdaptiveCpp" pip install -v .

# C++ only (driver + ctest smoke test)
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DSYCLNN_ONEMATH_ROOT=/opt/onemath -DSYCLNN_BUILD_PYTHON=OFF
cmake --build build -j8 && ctest --test-dir build
./build/train_bench --layers 784,1024,10 --samples 8192 --batch 256 --epochs 5 --dtype float --device gpu --profile
```

CMake options: `SYCLNN_SYCL_IMPL` (auto|dpcpp|adaptivecpp), `SYCLNN_TARGETS`,
`SYCLNN_ACPP_TARGETS`, `SYCLNN_ONEMATH_ROOT`, `SYCLNN_BLAS_BACKENDS` (for the
FetchContent build, default `mklcpu;netlib`), `SYCLNN_FAST_MATH`, `SYCLNN_BUILD_BENCH`,
`SYCLNN_BUILD_PYTHON`. The same variables are read from the environment by
`pip install` (scikit-build-core). Without a pre-built oneMath, CMake fetches and
builds oneMath v0.9 with the backends of `SYCLNN_BLAS_BACKENDS` (slow, once).

Selecting the BLAS backend at run time (`blas=mklcpu|netlib|generic`) needs the
compile-time dispatch entry points of oneMath: `SYCLNN_CT_BACKENDS` (default
`mklcpu;netlib;generic`) lists the backend libraries linked for that purpose.
GPU backends are deliberately not linked (the module would then need
`libcuda.so.1` / `libamdhip64.so` on machines without that GPU): `blas="auto"`
reaches cuBLAS / rocBLAS through oneMath's run-time loader. With AdaptiveCpp
builds also use `blas="auto"` on the CPU (the explicit NETLIB entry point fails
to JIT there).

## Tests

The parity suite lives in `fnn-bench/testkit` (`fnn-testkit`); this repository
runs it against `syclnn` plus a few smoke tests:

```sh
pip install "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit"
pytest                                  # default device, both dtypes
pytest --device gpu --dtype float --blas cublas --option memory=shared
pytest --device cpu --run-slow          # + MNIST
```

Status (2026-09-03): the parity suite passes in double and float on DPC++
(intel/llvm v7.1.0) `opencl:cpu` with MKLCPU and NETLIB/OpenBLAS, on the
AdaptiveCpp OpenMP host device and on the RX 7900 XTX (AdaptiveCpp + rocBLAS)
for every ablation switch, except `memory=host` on the AMD GPU (stale reads
through zero-copy host memory; reported as a platform result). NVIDIA runs are
tracked in `fnn-bench/docs/toolchains.md`.

## Container

```sh
podman build --memory=20g -t syclnn .
podman run --rm -it syclnn python -c "import syclnn; print(syclnn.devices())"
podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable syclnn pytest --device gpu
```

## License

LGPL-3.0-only. Copyright (C) 2025-2026 Antonio Napolitano, Suqi Chen.
