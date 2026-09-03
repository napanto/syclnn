# syclnn container: the library installed into the fnn-bench SYCL toolchain image.
#
#   podman build --memory=20g -t syclnn .                                  # CPU + NVIDIA (spir64 + sm_61 + sm_80)
#   podman build --memory=20g --build-arg SYCLNN_TARGETS=spir64 -t syclnn:cpu .
#   podman run --rm -it syclnn python -c "import syclnn; print(syclnn.devices())"
#   podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable syclnn ...
ARG BASE=ghcr.io/napanto/fnn-sycl:latest
FROM ${BASE}

ARG SYCLNN_TARGETS="spir64;nvidia_gpu_sm_61;nvidia_gpu_sm_80"
ARG SYCLNN_ONEMATH_ROOT=/opt/onemath
ENV SYCLNN_TARGETS=${SYCLNN_TARGETS} SYCLNN_ONEMATH_ROOT=${SYCLNN_ONEMATH_ROOT} CMAKE_BUILD_PARALLEL_LEVEL=8

COPY . /opt/src/syclnn
RUN pip install --no-cache-dir -v /opt/src/syclnn \
    && pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" \
    && python -c "import syclnn; print(syclnn.__version__, syclnn.build_info()); print(syclnn.devices())"

WORKDIR /opt/src/syclnn
CMD ["python"]
