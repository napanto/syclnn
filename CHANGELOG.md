# Changelog

All notable changes to syclnn. The project follows [Semantic Versioning](https://semver.org).

## [0.2.0] - 2026-09

Version 0.2 turns the ML-exam library into the SYCL reference implementation of
the *SYCL vs CUDA vs OpenMP* study (Accelerated Computing project). Every
behavioural change is behind an `Options` switch so that the benchmark can
measure it; the default configuration is the optimised one, the 0.1 behaviour is
always reachable (the switch that restores it is given in brackets).

### Added

- `Options.blas = "tiled"`: hand-written BLAS (a 16x16 tiled GEMM with local memory (nd_range),
  a row-per-work-item GEMV, reductions for asum/nrm2) for the "same kernel in
  the three programming models" comparison (E7 of the study).
- `Options` (device selection, BLAS backend selection, profiling, ablation
  switches) passed to every constructor; `Network::options()` reports it.
- Run-time device selection (`Options::device`: `default`, `cpu`, `gpu`,
  `index:N`, `cuda:0`, `opencl:cpu`, name substring) and `syclnn::devices()` /
  `syclnn.devices()` enumeration with vendor, backend, driver, memory, fp64.
- Run-time BLAS backend selection through oneMath's compile-time dispatch
  (`Options::blas = mklcpu | netlib | generic | cublas | rocblas`; `auto` keeps
  the run-time loader). `syclnn.build_info()["blas_backends"]` lists what is
  compiled in.
- Per-phase profiler (`Options::profile`, `Network::profile()`, `net.profile`
  dict): device nanoseconds for H2D, D2H, GEMM, activation, delta, bias
  gradient, update, loss, regularisation, plus host wall/wait time, launch and
  byte counters, per-epoch wall time. Fed by `event::get_profiling_info`;
  profiling is now opt-in (0.1 always enabled it).
- `weights` / `biases` properties (current values), `predict(..., batch_size)`,
  `parameter_count`, `device`, `blas_backend`, `reset_profile()`.
- Optional per-epoch shuffling (`Options::shuffle`, `shuffle_seed`) with a
  `std::mt19937` Fisher-Yates permutation reproduced by the test-kit.
- `bench/train_bench.cpp`: C++ driver without Python (also the ctest smoke test).
- Python: `syclnn.Network(..., dtype=, **options)` factory, `syclnn.Options`,
  `syclnn.make_options`; enums are Python `enum.Enum`s (pybind11 3 native enums).
- Test-suite: the shared `fnn-testkit` parity suite (`pytest` in the repo runs it
  against this backend), GitHub Actions CI on `opencl:cpu` with NETLIB and MKLCPU.
- AdaptiveCpp support (`-DSYCLNN_SYCL_IMPL=adaptivecpp`), used for AMD GPUs.

### Changed

- Namespace `syclnn::`; headers split into `syclnn/{config,activations,device,
  blas,profile,network}.hpp`.
- Build: CMake no longer forces `clang++`; any DPC++ (`icpx`, intel/llvm
  `clang++`) or AdaptiveCpp works. `SYCLNN_TARGETS` selects the offload targets
  (`spir64`, `nvidia_gpu_sm_61`, `nvidia_gpu_sm_80`, `amd_gpu_gfx1100`, ...);
  fat binaries by default, so one wheel runs on CPU and GPU. `find_package(oneMath)`
  is used when available (`SYCLNN_ONEMATH_ROOT`), FetchContent of oneMath v0.9
  otherwise (0.1 rebuilt oneMath v0.7 on every `pip install`).
- pybind11 3.1 (Python 3.12-3.14), scikit-build-core 1.x, Ubuntu 24.04 based
  container image built on `ghcr.io/napanto/fnn-sycl`.
- History snapshots: taken after every epoch only with `Options::record_history`
  (0.1 always did, with a full D2H copy and a `wait` per epoch); otherwise the
  history holds the initial and the final state. [`record_history=true`]
- `max_epochs == 0` raises `std::invalid_argument` (0.1 silently ran one epoch);
  `batch_size > num_samples` is clamped.
- Loss accumulated in double precision on the device (0.1: `float` for
  `Network<float>`).

### Optimisations (each with its ablation switch)

1. USM device memory + explicit copies instead of USM shared for every tensor
   [`memory=shared`]; `memory=host` gives zero-copy host memory.
2. Loss via `sycl::reduction` instead of one atomic add per work-item on a single
   scalar [`loss_reduction=false`].
3. Bias gradient via oneMath `gemv` with a ones vector instead of a serial loop
   per output neuron [`bias_gemv=false`].
4. The first GEMM reads the dataset slice directly; no per-batch input memcpy
   [`direct_input=false`].
5. Fine-grained event graph: each kernel depends only on its producers, so the
   gradient GEMM of layer *l* runs as soon as delta *l+1* exists and the updates
   overlap the remaining back-propagation [`fine_deps=false`]; empty "join"
   kernels replaced by multi-event dependencies [`join_kernels=true`].
6. In-order queue experiment [`queue=in_order`].
7. Kernels specialised on `ActivationType` at submit time instead of a `switch`
   inside the kernel [`specialized_kernels=false`]; derivatives from the stored
   output (sigma' = o(1-o), tanh' = 1-o^2, elu' = o+1) instead of recomputing
   the transcendental [`derivative_from_output=false`]; Adam bias-correction
   factors computed once per step on the host instead of `pow` per weight
   [`host_adam_correction=false`].
8. Optional `nd_range` launches with an explicit work-group size
   [`workgroup_size=N`]; all element-wise kernels use a flat index with the
   neuron index fastest (coalesced, column-major).
9. Persistent workspace across `train()`/`predict()` calls
   [`persistent_workspace=false`]; batched `predict`; pinned host staging for the
   dataset and the predictions [`pinned_host=false`].
10. Fused weight+bias update launch per layer (one launch instead of two).

### Fixed

- DPC++ CUDA backend: the queue is forced in-order whenever oneMath is the BLAS
  (`Options.queue` reports the effective value). The oneMath cuBLAS backend
  returns events that complete before its asynchronous cuBLAS work, so the
  out-of-order dependency graph raced on the GTX 1080 Ti (MNIST accuracy
  ~50 %); `blas=tiled` keeps the out-of-order queue.
- USM leak of the asum/nrm2 temporaries in the regularisation penalty (allocated
  per layer per epoch, never freed).
- Destructor could throw (`wait_and_throw`); it now swallows asynchronous errors.
- Missing RAII: any exception in `train()`/`predict()` leaked every device buffer;
  buffers are RAII now and the queue is drained before they are released.
- Initial-weight validation happened after part of the parameters had already
  been uploaded; a mismatch freed buffers with copies in flight.
- `predict()` allocated delta/gradient/loss buffers it never used, sized for the
  whole dataset, on every call.
- Unused lambda in `Regularization` is ignored consistently (L1 ignores lambda2,
  L2 ignores lambda1), biases are never regularised - now covered by tests.

## [0.1.0] - 2025-05-19

Initial release for the Machine Learning exam project.
