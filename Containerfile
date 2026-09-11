# syclnn installed into a toolchain image of the study (fnn-bench/containers). Variants by build argument:
#   default        BASE=ghcr.io/napanto/fnn-sycl:latest   DPC++ (the image's clang), spir64 + sm_61 + sm_80, oneMath /opt/onemath
#   acpp-rocm      BASE=ghcr.io/napanto/fnn-rocm:latest   BUILD_CC=clang-18 BUILD_CXX=acpp ONEMATH_ROOT=/opt/fnn-rocm/onemath CT_BACKENDS=netlib
#   acpp-cuda      BASE=ghcr.io/napanto/fnn-acpp-cuda:latest  BUILD_CC=clang-18 BUILD_CXX=acpp ONEMATH_ROOT=/opt/onemath-acpp CT_BACKENDS=netlib
# (the AdaptiveCpp bases set SYCLNN_SYCL_IMPL and AdaptiveCpp_DIR themselves)
ARG BASE=ghcr.io/napanto/fnn-sycl:latest
FROM ${BASE}
ARG IMAGE_NAME=syclnn:latest
ARG IMAGE_BUILT=unknown
ARG BUILD_CC=clang
ARG BUILD_CXX=clang++
ARG SYCL_TARGETS="spir64;nvidia_gpu_sm_61;nvidia_gpu_sm_80"
ARG ONEMATH_ROOT=/opt/onemath
ARG CT_BACKENDS=""
ENV CC=${BUILD_CC} CXX=${BUILD_CXX} SYCLNN_TARGETS=${SYCL_TARGETS} SYCLNN_ONEMATH_ROOT=${ONEMATH_ROOT} CMAKE_BUILD_PARALLEL_LEVEL=8
COPY . /opt/src/syclnn
RUN if [ -n "${CT_BACKENDS}" ]; then export SYCLNN_CT_BACKENDS="${CT_BACKENDS}"; fi \
    && pip install --no-cache-dir -v /opt/src/syclnn \
    && pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" \
    && python -c "import syclnn; print(syclnn.__version__, syclnn.build_info()); print(syclnn.devices())"
ENV FNN_IMAGE=${IMAGE_NAME} FNN_IMAGE_BUILT=${IMAGE_BUILT}
LABEL org.opencontainers.image.source=https://github.com/napanto/syclnn fnn.image="${IMAGE_NAME}" fnn.image.built="${IMAGE_BUILT}"
WORKDIR /opt/src/syclnn
CMD ["python"]
