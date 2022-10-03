# at config level import appropriate kernels.
from ipie.config import config
if config.get_option('use_gpu'):
    from .gpu.exchange import exchange_reduction
    from .gpu import wicks
else:
    exchange_reduction = None
    from .cpu import wicks
