# syclnn

SYCL-accelerated feed-forward neural network library.

Uses Intel's DPC++ compiler and UXL Foundation's [oneMath library](https://github.com/uxlfoundation/oneMath).

Python bindings are provided.

## Requirements
- [CMake](https://cmake.org/download/)
- [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- (optional, for NVIDIA GPUs) [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [oneAPI for NVIDIA® GPUs](https://developer.codeplay.com/products/oneapi/nvidia/home/)

## Installation and usage
```
pip install .

# for NVIDIA GPU support
pip install . --config-settings=cmake.args=-DENABLE_NVIDIA_GPU=ON
```

Then you can import the module in Python
```
Python 3.12.5 (main, Apr  2 2025, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import syclnn
>>> dir(syclnn)
['ActivationType', 'AdaGrad', 'Adam', 'AdaptiveLearningRate_double', 'AdaptiveLearningRate_float', 'AdaptiveLearningStrategy_double', 'AdaptiveLearningStrategy_float', 'BackPropagation', 'Classical', 'Constant', 'Disabled', 'ELU', 'ElasticNet', 'L1', 'L2', 'LayerDescription', 'LeakyReLU', 'LinearDecay', 'MaxEpochs', 'MinError', 'MinErrorChange', 'MomentumConfig_double', 'MomentumConfig_float', 'MomentumType_double', 'MomentumType_float', 'Network_double', 'Network_float', 'RMSProp', 'ReLU', 'RegularizationType_double', 'RegularizationType_float', 'Regularization_double', 'Regularization_float', 'Sigmoid', 'Standard', 'StopCriteriaType_double', 'StopCriteriaType_float', 'StopCriteria_double', 'StopCriteria_float', 'Tanh', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'get_sycl_devices']
```

## Using a container

The container uploaded to GitHub's registry contains a Python environment with the syclnn module already built and installed.

### Running

```
docker run --rm --name syclnn-env -it ghcr.io/napaalm/syclnn:latest

# for NVIDIA GPU support
docker run --rm --gpus all --name syclnn-env -it ghcr.io/napaalm/syclnn:nvidia
```

Run `python3.12` inside this container.

#### Running with Podman
```
podman run --rm --name syclnn-env -it syclnn:latest

# for NVIDIA GPU support
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable --name syclnn-env -it syclnn:nvidia
```

### Build it yourself
```
docker build -t syclnn .

# for NVIDIA GPU support
docker build -t syclnn . --build-arg ENABLE_NVIDIA=1
```
