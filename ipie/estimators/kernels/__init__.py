# at config level import appropriate kernels.
from ipie.config import config
if config.get_option('use_gpu'):
    from .gpu.exchange import exchange_reduction
    import .gpu.wicks as wicks
else:
    exchange_reduction = None
    import .cpu.wicks as wicks
