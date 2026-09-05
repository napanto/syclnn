"""
syclnn: SYCL feed-forward neural network (oneMath BLAS + hand-written kernels)
"""
from __future__ import annotations
import collections.abc
import enum
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ActivationType', 'AdaptiveLearningRate_double', 'AdaptiveLearningRate_float', 'AdaptiveLearningStrategy_double', 'AdaptiveLearningStrategy_float', 'BackPropagation', 'Disabled', 'ELU', 'LayerDescription', 'LeakyReLU', 'MemoryKind', 'MomentumConfig_double', 'MomentumConfig_float', 'MomentumType_double', 'MomentumType_float', 'Network_double', 'Network_float', 'Options', 'QueueOrder', 'ReLU', 'RegularizationType_double', 'RegularizationType_float', 'Regularization_double', 'Regularization_float', 'Sigmoid', 'Standard', 'StopCriteriaType_double', 'StopCriteriaType_float', 'StopCriteria_double', 'StopCriteria_float', 'Tanh', 'build_info', 'devices', 'get_sycl_devices']
class ActivationType(enum.Enum):
    Disabled: typing.ClassVar[ActivationType]  # value = <ActivationType.Disabled: 0>
    ELU: typing.ClassVar[ActivationType]  # value = <ActivationType.ELU: 5>
    LeakyReLU: typing.ClassVar[ActivationType]  # value = <ActivationType.LeakyReLU: 4>
    ReLU: typing.ClassVar[ActivationType]  # value = <ActivationType.ReLU: 3>
    Sigmoid: typing.ClassVar[ActivationType]  # value = <ActivationType.Sigmoid: 1>
    Tanh: typing.ClassVar[ActivationType]  # value = <ActivationType.Tanh: 2>
class AdaptiveLearningRate_double:
    strategy: AdaptiveLearningStrategy_double
    def __init__(self, strategy: AdaptiveLearningStrategy_double = ..., epsilon: typing.SupportsFloat | typing.SupportsIndex = 1e-08, beta1: typing.SupportsFloat | typing.SupportsIndex = 0.9, beta2: typing.SupportsFloat | typing.SupportsIndex = 0.999, final_learning_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0001) -> None:
        ...
    @property
    def beta1(self) -> float:
        ...
    @beta1.setter
    def beta1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta2(self) -> float:
        ...
    @beta2.setter
    def beta2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def final_learning_rate(self) -> float:
        ...
    @final_learning_rate.setter
    def final_learning_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class AdaptiveLearningRate_float:
    strategy: AdaptiveLearningStrategy_float
    def __init__(self, strategy: AdaptiveLearningStrategy_float = ..., epsilon: typing.SupportsFloat | typing.SupportsIndex = 9.99999993922529e-09, beta1: typing.SupportsFloat | typing.SupportsIndex = 0.8999999761581421, beta2: typing.SupportsFloat | typing.SupportsIndex = 0.9990000128746033, final_learning_rate: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-05) -> None:
        ...
    @property
    def beta1(self) -> float:
        ...
    @beta1.setter
    def beta1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta2(self) -> float:
        ...
    @beta2.setter
    def beta2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def final_learning_rate(self) -> float:
        ...
    @final_learning_rate.setter
    def final_learning_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class AdaptiveLearningStrategy_double(enum.Enum):
    AdaGrad: typing.ClassVar[AdaptiveLearningStrategy_double]  # value = <AdaptiveLearningStrategy_double.AdaGrad: 2>
    Adam: typing.ClassVar[AdaptiveLearningStrategy_double]  # value = <AdaptiveLearningStrategy_double.Adam: 4>
    Constant: typing.ClassVar[AdaptiveLearningStrategy_double]  # value = <AdaptiveLearningStrategy_double.Constant: 0>
    LinearDecay: typing.ClassVar[AdaptiveLearningStrategy_double]  # value = <AdaptiveLearningStrategy_double.LinearDecay: 1>
    RMSProp: typing.ClassVar[AdaptiveLearningStrategy_double]  # value = <AdaptiveLearningStrategy_double.RMSProp: 3>
class AdaptiveLearningStrategy_float(enum.Enum):
    AdaGrad: typing.ClassVar[AdaptiveLearningStrategy_float]  # value = <AdaptiveLearningStrategy_float.AdaGrad: 2>
    Adam: typing.ClassVar[AdaptiveLearningStrategy_float]  # value = <AdaptiveLearningStrategy_float.Adam: 4>
    Constant: typing.ClassVar[AdaptiveLearningStrategy_float]  # value = <AdaptiveLearningStrategy_float.Constant: 0>
    LinearDecay: typing.ClassVar[AdaptiveLearningStrategy_float]  # value = <AdaptiveLearningStrategy_float.LinearDecay: 1>
    RMSProp: typing.ClassVar[AdaptiveLearningStrategy_float]  # value = <AdaptiveLearningStrategy_float.RMSProp: 3>
class BackPropagation(enum.Enum):
    Standard: typing.ClassVar[BackPropagation]  # value = <BackPropagation.Standard: 0>
class LayerDescription:
    activation: ActivationType
    def __init__(self, neurons: typing.SupportsInt | typing.SupportsIndex, activation: ActivationType = ...) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def neurons(self) -> int:
        ...
    @neurons.setter
    def neurons(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class MemoryKind(enum.Enum):
    Device: typing.ClassVar[MemoryKind]  # value = <MemoryKind.Device: 0>
    Host: typing.ClassVar[MemoryKind]  # value = <MemoryKind.Host: 2>
    Shared: typing.ClassVar[MemoryKind]  # value = <MemoryKind.Shared: 1>
class MomentumConfig_double:
    type: MomentumType_double
    def __init__(self, type: MomentumType_double = ..., momentum_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    @property
    def momentum_rate(self) -> float:
        ...
    @momentum_rate.setter
    def momentum_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MomentumConfig_float:
    type: MomentumType_float
    def __init__(self, type: MomentumType_float = ..., momentum_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    @property
    def momentum_rate(self) -> float:
        ...
    @momentum_rate.setter
    def momentum_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MomentumType_double(enum.Enum):
    Classical: typing.ClassVar[MomentumType_double]  # value = <MomentumType_double.Classical: 1>
    Disabled: typing.ClassVar[MomentumType_double]  # value = <MomentumType_double.Disabled: 0>
class MomentumType_float(enum.Enum):
    Classical: typing.ClassVar[MomentumType_float]  # value = <MomentumType_float.Classical: 1>
    Disabled: typing.ClassVar[MomentumType_float]  # value = <MomentumType_float.Disabled: 0>
class Network_double:
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_double = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_double = ..., stop_criteria: StopCriteria_double = ..., momentum: MomentumConfig_double = ..., options: Options = ...) -> None:
        """
        Random initialisation seeded from the clock.
        """
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_double = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_double = ..., stop_criteria: StopCriteria_double = ..., momentum: MomentumConfig_double = ..., seed: typing.SupportsInt | typing.SupportsIndex, options: Options = ...) -> None:
        """
        Random initialisation in [-0.1, 0.1] from std::mt19937(seed).
        """
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_double = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_double = ..., stop_criteria: StopCriteria_double = ..., momentum: MomentumConfig_double = ..., initial_weights: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64]], initial_biases: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64]], options: Options = ...) -> None:
        """
        Explicit initial weights (flattened column-major (n_out, n_in)) and biases.
        """
    def predict(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], n_samples: typing.SupportsInt | typing.SupportsIndex, batch_size: typing.SupportsInt | typing.SupportsIndex = 0) -> numpy.typing.NDArray[numpy.float64]:
        """
        Forward pass; returns flattened outputs (n_samples x n_out). batch_size 0 = one batch.
        """
    def reset_profile(self) -> None:
        ...
    def set_weights_biases(self, weights: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64]], biases: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64]]) -> None:
        """
        Replace all weights / biases and reset the history.
        """
    def train(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], n_samples: typing.SupportsInt | typing.SupportsIndex, batch_size: typing.SupportsInt | typing.SupportsIndex, max_epochs: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
        """
        Train the network.
        
        x, y : flattened inputs (n_samples x n_in) and targets (n_samples x n_out), sample-major.
        Returns the total loss (mean data loss + regularisation penalty) of every epoch run.
        """
    @property
    def biases(self) -> list:
        """
        Current biases, one array per layer.
        """
    @property
    def blas_backend(self) -> str:
        ...
    @property
    def device(self) -> dict:
        ...
    @property
    def device_name(self) -> str:
        ...
    @property
    def layers(self) -> list[LayerDescription]:
        ...
    @property
    def options(self) -> Options:
        ...
    @property
    def parameter_count(self) -> int:
        ...
    @property
    def profile(self) -> dict:
        """
        Per-phase device times (ns) and counters; all zero unless options.profile.
        """
    @property
    def weights(self) -> list:
        """
        Current weights, one flattened column-major (n_out, n_in) array per layer.
        """
    @property
    def weights_biases(self) -> tuple[list, list]:
        """
        Snapshots (weights_history, biases_history): [snapshot][layer] flattened column-major arrays. Snapshot 0 is the initial state; one per epoch with options.record_history, else the final state.
        """
class Network_float:
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_float = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_float = ..., stop_criteria: StopCriteria_float = ..., momentum: MomentumConfig_float = ..., options: Options = ...) -> None:
        """
        Random initialisation seeded from the clock.
        """
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_float = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_float = ..., stop_criteria: StopCriteria_float = ..., momentum: MomentumConfig_float = ..., seed: typing.SupportsInt | typing.SupportsIndex, options: Options = ...) -> None:
        """
        Random initialisation in [-0.1, 0.1] from std::mt19937(seed).
        """
    @typing.overload
    def __init__(self, layers: collections.abc.Sequence[LayerDescription], learning_rate: typing.SupportsFloat | typing.SupportsIndex, regularization: Regularization_float = ..., backpropagation: BackPropagation = ..., adaptive_learning_rate: AdaptiveLearningRate_float = ..., stop_criteria: StopCriteria_float = ..., momentum: MomentumConfig_float = ..., initial_weights: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float32]], initial_biases: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float32]], options: Options = ...) -> None:
        """
        Explicit initial weights (flattened column-major (n_out, n_in)) and biases.
        """
    def predict(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], n_samples: typing.SupportsInt | typing.SupportsIndex, batch_size: typing.SupportsInt | typing.SupportsIndex = 0) -> numpy.typing.NDArray[numpy.float32]:
        """
        Forward pass; returns flattened outputs (n_samples x n_out). batch_size 0 = one batch.
        """
    def reset_profile(self) -> None:
        ...
    def set_weights_biases(self, weights: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float32]], biases: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float32]]) -> None:
        """
        Replace all weights / biases and reset the history.
        """
    def train(self, x: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], n_samples: typing.SupportsInt | typing.SupportsIndex, batch_size: typing.SupportsInt | typing.SupportsIndex, max_epochs: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float32]:
        """
        Train the network.
        
        x, y : flattened inputs (n_samples x n_in) and targets (n_samples x n_out), sample-major.
        Returns the total loss (mean data loss + regularisation penalty) of every epoch run.
        """
    @property
    def biases(self) -> list:
        """
        Current biases, one array per layer.
        """
    @property
    def blas_backend(self) -> str:
        ...
    @property
    def device(self) -> dict:
        ...
    @property
    def device_name(self) -> str:
        ...
    @property
    def layers(self) -> list[LayerDescription]:
        ...
    @property
    def options(self) -> Options:
        ...
    @property
    def parameter_count(self) -> int:
        ...
    @property
    def profile(self) -> dict:
        """
        Per-phase device times (ns) and counters; all zero unless options.profile.
        """
    @property
    def weights(self) -> list:
        """
        Current weights, one flattened column-major (n_out, n_in) array per layer.
        """
    @property
    def weights_biases(self) -> tuple[list, list]:
        """
        Snapshots (weights_history, biases_history): [snapshot][layer] flattened column-major arrays. Snapshot 0 is the initial state; one per epoch with options.record_history, else the final state.
        """
class Options:
    """
    Runtime options: device, BLAS backend, profiling, ablation switches.
    """
    bias_gemv: bool
    blas: str
    derivative_from_output: bool
    device: str
    direct_input: bool
    fast_math: bool
    sync_every: int
    blas_queue: str
    fine_deps: bool
    host_adam_correction: bool
    join_kernels: bool
    loss_reduction: bool
    persistent_workspace: bool
    pinned_host: bool
    profile: bool
    record_history: bool
    shuffle: bool
    specialized_kernels: bool
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def memory(self) -> MemoryKind:
        ...
    @memory.setter
    def memory(self, arg1: typing.Any) -> None:
        ...
    @property
    def queue(self) -> QueueOrder:
        ...
    @queue.setter
    def queue(self, arg1: typing.Any) -> None:
        ...
    @property
    def shuffle_seed(self) -> int:
        ...
    @shuffle_seed.setter
    def shuffle_seed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def streams(self) -> int:
        ...
    @streams.setter
    def streams(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def workgroup_size(self) -> int:
        ...
    @workgroup_size.setter
    def workgroup_size(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class QueueOrder(enum.Enum):
    Graph: typing.ClassVar[QueueOrder]  # value = <QueueOrder.Graph: 2>
    InOrder: typing.ClassVar[QueueOrder]  # value = <QueueOrder.InOrder: 1>
    OutOfOrder: typing.ClassVar[QueueOrder]  # value = <QueueOrder.OutOfOrder: 0>
class RegularizationType_double(enum.Enum):
    Disabled: typing.ClassVar[RegularizationType_double]  # value = <RegularizationType_double.Disabled: 0>
    ElasticNet: typing.ClassVar[RegularizationType_double]  # value = <RegularizationType_double.ElasticNet: 3>
    L1: typing.ClassVar[RegularizationType_double]  # value = <RegularizationType_double.L1: 1>
    L2: typing.ClassVar[RegularizationType_double]  # value = <RegularizationType_double.L2: 2>
class RegularizationType_float(enum.Enum):
    Disabled: typing.ClassVar[RegularizationType_float]  # value = <RegularizationType_float.Disabled: 0>
    ElasticNet: typing.ClassVar[RegularizationType_float]  # value = <RegularizationType_float.ElasticNet: 3>
    L1: typing.ClassVar[RegularizationType_float]  # value = <RegularizationType_float.L1: 1>
    L2: typing.ClassVar[RegularizationType_float]  # value = <RegularizationType_float.L2: 2>
class Regularization_double:
    type: RegularizationType_double
    def __init__(self, type: RegularizationType_double = ..., lambda1: typing.SupportsFloat | typing.SupportsIndex = 0.0, lambda2: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    @property
    def lambda1(self) -> float:
        ...
    @lambda1.setter
    def lambda1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lambda2(self) -> float:
        ...
    @lambda2.setter
    def lambda2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Regularization_float:
    type: RegularizationType_float
    def __init__(self, type: RegularizationType_float = ..., lambda1: typing.SupportsFloat | typing.SupportsIndex = 0.0, lambda2: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
        ...
    @property
    def lambda1(self) -> float:
        ...
    @lambda1.setter
    def lambda1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def lambda2(self) -> float:
        ...
    @lambda2.setter
    def lambda2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class StopCriteriaType_double(enum.Enum):
    MaxEpochs: typing.ClassVar[StopCriteriaType_double]  # value = <StopCriteriaType_double.MaxEpochs: 0>
    MinError: typing.ClassVar[StopCriteriaType_double]  # value = <StopCriteriaType_double.MinError: 1>
    MinErrorChange: typing.ClassVar[StopCriteriaType_double]  # value = <StopCriteriaType_double.MinErrorChange: 2>
class StopCriteriaType_float(enum.Enum):
    MaxEpochs: typing.ClassVar[StopCriteriaType_float]  # value = <StopCriteriaType_float.MaxEpochs: 0>
    MinError: typing.ClassVar[StopCriteriaType_float]  # value = <StopCriteriaType_float.MinError: 1>
    MinErrorChange: typing.ClassVar[StopCriteriaType_float]  # value = <StopCriteriaType_float.MinErrorChange: 2>
class StopCriteria_double:
    type: StopCriteriaType_double
    def __init__(self, type: StopCriteriaType_double = ..., threshold: typing.SupportsFloat | typing.SupportsIndex = 0.0001) -> None:
        ...
    @property
    def threshold(self) -> float:
        ...
    @threshold.setter
    def threshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class StopCriteria_float:
    type: StopCriteriaType_float
    def __init__(self, type: StopCriteriaType_float = ..., threshold: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-05) -> None:
        ...
    @property
    def threshold(self) -> float:
        ...
    @threshold.setter
    def threshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def build_info() -> dict:
    """
    How this module was built: compiler, flags, SYCL targets, oneMath backends.
    """
def devices() -> list:
    """
    All SYCL devices visible to this build, as dicts (index, name, type, backend, ...).
    """
def get_sycl_devices() -> list[str]:
    """
    0.1 compatibility: names of all visible SYCL devices.
    """
Disabled: ActivationType  # value = <ActivationType.Disabled: 0>
ELU: ActivationType  # value = <ActivationType.ELU: 5>
LeakyReLU: ActivationType  # value = <ActivationType.LeakyReLU: 4>
ReLU: ActivationType  # value = <ActivationType.ReLU: 3>
Sigmoid: ActivationType  # value = <ActivationType.Sigmoid: 1>
Standard: BackPropagation  # value = <BackPropagation.Standard: 0>
Tanh: ActivationType  # value = <ActivationType.Tanh: 2>
__version__: str = '0.2.0'
