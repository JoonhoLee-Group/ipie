import pytest

@pytest.fixture
def gpu_env(pytestconfig):
    markers_arg = pytestconfig.getoption('-m')
    print("markers arg: ", markers_arg, markers_arg == 'gpu')
    if markers_arg == 'gpu':
        pass
        # from ipie.config import purge_ipie_modules, config
        # purge_ipie_modules()
        # import sys
        # print([s  for s in sys.modules.keys() if 'ipie' in s])
        # config.update_option('use_gpu', True)
        # purge_ipie_modules()
        # config.update_option('use_gpu', True)
        # from ipie.utils import backend
        # import importlib
        # importlib.reload(backend)
        # from ipie.utils.backend import arraylib as xp
