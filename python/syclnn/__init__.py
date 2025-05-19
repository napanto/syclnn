from importlib import import_module as _imp
_syclnn = _imp("._syclnn", package=__name__)   # eager import
globals().update(_syclnn.__dict__)             # re-export C++ API
del _imp, _syclnn
