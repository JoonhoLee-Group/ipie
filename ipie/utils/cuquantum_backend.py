from ipie.utils.backend import arraylib as xp


def _missing_cuquantum(*args, **kwargs):
    raise ImportError(
        "cuquantum>=26.1 is required for k-point cuTensorNet contractions. "
        "Install cuquantum-python to enable this code path."
    )


class _CutensornetFallback:
    @staticmethod
    def create():
        return None

    @staticmethod
    def destroy(_handle):
        return None

    @staticmethod
    def contract(subscripts, *operands, **kwargs):
        kwargs.pop("options", None)
        kwargs.pop("handle", None)
        return xp.einsum(subscripts, *operands, **kwargs)


class _NetworkOptionsFallback:
    def __init__(self, *args, **kwargs):
        pass


class _CutensornetMissing:
    create = staticmethod(_missing_cuquantum)
    destroy = staticmethod(lambda _handle: None)


class _NetworkOptionsMissing:
    def __init__(self, *args, **kwargs):
        _missing_cuquantum()


def _contract_missing(subscripts, *operands, **kwargs):
    _missing_cuquantum()


class _CuTensorNetCompat:
    def __init__(self, backend, contract_fn):
        self._backend = backend
        self.contract = contract_fn

    def create(self):
        return self._backend.create()

    def destroy(self, handle):
        return self._backend.destroy(handle)

    def __getattr__(self, name):
        return getattr(self._backend, name)


try:
    from cuquantum.bindings import cutensornet as _cutensornet
    from cuquantum.tensornet import NetworkOptions as _NetworkOptions
    from cuquantum.tensornet import contract as _contract

    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False

    _cutensornet = _CutensornetFallback()
    _NetworkOptions = _NetworkOptionsFallback
    _contract = _CutensornetFallback.contract


cutensornet_optional = _CuTensorNetCompat(_cutensornet, _contract)
NetworkOptions_optional = _NetworkOptions
contract_optional = _contract


if HAS_CUQUANTUM:
    cutensornet_required = _CuTensorNetCompat(_cutensornet, _contract)
    NetworkOptions_required = _NetworkOptions
    contract_required = _contract
else:
    cutensornet_required = _CutensornetMissing()
    NetworkOptions_required = _NetworkOptionsMissing
    contract_required = _contract_missing
