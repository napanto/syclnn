# Use the official AlmaLinux 9 base image
FROM docker.io/library/almalinux:9

ARG ENABLE_NVIDIA=0

# Install necessary utilities and dependencies, and development tools
RUN dnf update -y && \
    dnf install -y --allowerasing dnf-plugins-core curl wget cmake pkgconfig procps && \
    dnf groupinstall -y "Development Tools" && \
    dnf clean all && \
    rm -rf /var/cache/yum

# -------------------------------
# Repository configuration
# -------------------------------

# Add the Intel oneAPI repository
RUN tee /etc/yum.repos.d/oneAPI.repo <<EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
priority=50
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

# Add the NVIDIA CUDA repository
RUN tee /etc/yum.repos.d/CUDA.repo <<EOF
[CUDA]
name=CUDA
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/D42D0685.pub
EOF

# -------------------------------
# Install the toolkit packages
# -------------------------------

RUN dnf update -y && \
    dnf install -y intel-oneapi-base-toolkit && \
    dnf clean all && \
    rm -rf /var/cache/yum

RUN if [ "$ENABLE_NVIDIA" -eq "1" ] ; then dnf install -y cuda-toolkit && \
    dnf clean all && \
    rm -rf /var/cache/yum ; fi

# -------------------------------
# Install codeplay plugins
# -------------------------------

# oneAPI for Nvidia GPUs
RUN if [ "$ENABLE_NVIDIA" -eq "1" ] ; then curl -L -o /tmp/oneapi-for-nvidia-gpus-linux.sh \
    "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&filters[]=linux" && \
    chmod +x /tmp/oneapi-for-nvidia-gpus-linux.sh && \
    /tmp/oneapi-for-nvidia-gpus-linux.sh -y && \
    rm /tmp/oneapi-for-nvidia-gpus-linux.sh ; fi

# -------------------------------
# Entrypoint
# -------------------------------
RUN tee /entrypoint.sh <<'EOF'
#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LIBRARY_PATH
export CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
source /opt/intel/oneapi/setvars.sh --include-intel-llvm
exec "$@"
EOF

RUN chmod +x /entrypoint.sh

# -------------------------------
# Python
# -------------------------------

ENV PIP_ROOT_USER_ACTION=ignore

RUN dnf install -y python3.12 python3.12-devel python3.12-pip && \
    dnf clean all && \
    rm -rf /var/cache/yum && \
    python3.12 -m pip install --no-cache-dir numpy

COPY . /syclnn

WORKDIR /syclnn
RUN /entrypoint.sh python3.12 -m pip install --no-cache-dir . \
    --config-settings=cmake.args=-DENABLE_NVIDIA_GPU:BOOL=$ENABLE_NVIDIA

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
