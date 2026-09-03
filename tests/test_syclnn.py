"""Backend-specific smoke tests; the full parity suite is collected from fnn_testkit
(see tests/conftest.py and pytest.ini)."""

import numpy as np
import pytest

syclnn = pytest.importorskip("syclnn")


def test_import_and_devices():
    assert syclnn.__version__.startswith("0.2")
    devs = syclnn.devices()
    assert devs and all("name" in d for d in devs)
    assert syclnn.get_sycl_devices() == [d["name"] for d in devs]
    info = syclnn.build_info()
    assert "blas_backends" in info and isinstance(info["blas_backends"], list)


def test_factory_and_options():
    layers = [syclnn.LayerDescription(3, syclnn.ActivationType.Disabled),
              syclnn.LayerDescription(2, syclnn.ActivationType.Sigmoid)]
    net = syclnn.Network(layers, 0.1, dtype="float", profile=True, seed=1)
    assert net.options.profile is True
    X = np.random.default_rng(0).uniform(-1, 1, (8, 3)).astype(np.float32)
    Y = np.zeros((8, 2), dtype=np.float32)
    losses = net.train(X.ravel(), Y.ravel(), 8, 4, 2)
    assert losses.shape == (2,)
    assert net.profile["epochs"] == 2 and net.profile["batches"] == 4
    out = net.predict(X.ravel(), 8)
    assert out.shape == (16,)
    with pytest.raises(AttributeError):
        syclnn.make_options(no_such=1)


def test_blas_backend_selection_errors():
    layers = [syclnn.LayerDescription(2), syclnn.LayerDescription(1, syclnn.ActivationType.Tanh)]
    with pytest.raises(ValueError):
        syclnn.Network(layers, 0.1, blas="no-such-backend", seed=1)
